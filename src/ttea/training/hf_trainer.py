from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, PlatformConfig
from ..evaluation import BenchmarkEvaluator
from ..exceptions import TrainingError
from ..integrations import import_datasets, import_torch, import_transformers
from ..persistence import list_checkpoint_directories, persist_checkpoint_index
from ..types import TaskExecutionResult, TaskSpec
from ..utils import ensure_directory


@dataclass(slots=True)
class TrainingArtifact:
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_index_path: str | None


class HFTrainingService:
    def __init__(self, platform_config: PlatformConfig, experiment_config: ExperimentConfig) -> None:
        self.platform_config = platform_config
        self.experiment_config = experiment_config
        self._transformers = import_transformers()
        self._datasets = import_datasets()
        self._torch = import_torch()
        self._evaluator = BenchmarkEvaluator(experiment_config)
        self._require_dependencies()

    def train(self, train_tasks: list[TaskSpec], eval_tasks: list[TaskSpec], output_dir: str | Path) -> TrainingArtifact:
        if not train_tasks:
            raise TrainingError("Training task set is empty.")
        if not eval_tasks:
            raise TrainingError("Evaluation task set is empty.")
        output_root = ensure_directory(output_dir)
        tokenizer = self._load_tokenizer()
        if self.experiment_config.training.task_type == "sequence_classification":
            return self._train_sequence_classification(train_tasks, eval_tasks, tokenizer, output_root)
        return self._train_seq2seq(train_tasks, eval_tasks, tokenizer, output_root)

    def _require_dependencies(self) -> None:
        if self._transformers is None or self._datasets is None:
            raise TrainingError("transformers and datasets must be installed to build the training pipeline.")

    def _load_tokenizer(self):
        try:
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                self.platform_config.models.tokenizer_name_or_path,
                local_files_only=self.platform_config.models.local_files_only,
                trust_remote_code=self.platform_config.models.trust_remote_code,
            )
        except Exception as exc:
            raise TrainingError(
                f"Unable to load tokenizer from {self.platform_config.models.tokenizer_name_or_path}: {exc}"
            ) from exc
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_seq2seq_model(self):
        try:
            model = self._transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.platform_config.models.model_name_or_path,
                local_files_only=self.platform_config.models.local_files_only,
                trust_remote_code=self.platform_config.models.trust_remote_code,
            )
        except Exception as exc:
            raise TrainingError(
                f"Unable to load a seq2seq model from {self.platform_config.models.model_name_or_path}: {exc}"
            ) from exc
        return model

    def _load_sequence_classification_model(self, label_count: int, label2id: dict[str, int], id2label: dict[int, str]):
        try:
            model = self._transformers.AutoModelForSequenceClassification.from_pretrained(
                self.platform_config.models.model_name_or_path,
                local_files_only=self.platform_config.models.local_files_only,
                trust_remote_code=self.platform_config.models.trust_remote_code,
                num_labels=label_count,
                label2id=label2id,
                id2label=id2label,
            )
        except Exception as exc:
            raise TrainingError(
                f"Unable to load a sequence classification model from {self.platform_config.models.model_name_or_path}: {exc}"
            ) from exc
        return model

    def _train_seq2seq(self, train_tasks: list[TaskSpec], eval_tasks: list[TaskSpec], tokenizer, output_dir: Path) -> TrainingArtifact:
        model = self._load_seq2seq_model()
        datasets = self._build_dataset_bundle(train_tasks, eval_tasks, include_labels=False)
        train_dataset = self._datasets.Dataset.from_list(datasets["train"])
        eval_dataset = self._datasets.Dataset.from_list(datasets["eval"])

        def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
            model_inputs = tokenizer(
                batch["input_text"],
                truncation=True,
                max_length=self.platform_config.models.max_prompt_tokens,
            )
            labels = tokenizer(
                text_target=batch["target_text"],
                truncation=True,
                max_length=self.platform_config.models.generation.get("max_new_tokens", 128),
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
        tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)
        training_args = self._transformers.Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.experiment_config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.experiment_config.training.per_device_eval_batch_size,
            learning_rate=self.experiment_config.training.learning_rate,
            weight_decay=self.experiment_config.training.weight_decay,
            num_train_epochs=self.experiment_config.training.num_train_epochs,
            warmup_ratio=self.experiment_config.training.warmup_ratio,
            gradient_accumulation_steps=self.experiment_config.training.gradient_accumulation_steps,
            logging_steps=self.experiment_config.training.logging_steps,
            save_steps=self.experiment_config.training.save_steps,
            predict_with_generate=True,
            generation_max_length=self.platform_config.models.generation.get("max_new_tokens", 128),
            fp16=self.experiment_config.training.fp16,
            bf16=self.experiment_config.training.bf16,
            evaluation_strategy="steps",
            save_strategy="steps",
            report_to=[],
            logging_dir=str(output_dir / "logs"),
            load_best_model_at_end=False,
        )
        data_collator = self._transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = self._transformers.Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._build_generative_metrics(tokenizer, eval_tasks),
        )
        trainer.train()
        evaluation = trainer.evaluate()
        final_model_dir = ensure_directory(output_dir / "final_model")
        trainer.save_model(str(final_model_dir))
        history = [dict(row) for row in trainer.state.log_history]
        checkpoint_index_path = self._persist_checkpoints(output_dir, {"task_type": "seq2seq", "evaluation": evaluation})
        return TrainingArtifact(
            summary={
                "task_type": self.experiment_config.training.task_type,
                "evaluation": evaluation,
                "output_dir": str(output_dir),
                "final_model_dir": str(final_model_dir),
            },
            history=history,
            checkpoint_index_path=checkpoint_index_path,
        )

    def _train_sequence_classification(
        self,
        train_tasks: list[TaskSpec],
        eval_tasks: list[TaskSpec],
        tokenizer,
        output_dir: Path,
    ) -> TrainingArtifact:
        train_examples = self._build_dataset_bundle(train_tasks, eval_tasks, include_labels=True)
        labels = sorted({str(example["target_text"]) for example in train_examples["train"] + train_examples["eval"]})
        label2id = {label: index for index, label in enumerate(labels)}
        id2label = {index: label for label, index in label2id.items()}
        model = self._load_sequence_classification_model(len(labels), label2id, id2label)
        train_dataset = self._datasets.Dataset.from_list(train_examples["train"])
        eval_dataset = self._datasets.Dataset.from_list(train_examples["eval"])

        def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
            encoded = tokenizer(
                batch["input_text"],
                truncation=True,
                max_length=self.platform_config.models.max_prompt_tokens,
            )
            encoded["labels"] = [label2id[str(label)] for label in batch["target_text"]]
            return encoded

        tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
        tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)
        training_args = self._transformers.TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.experiment_config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.experiment_config.training.per_device_eval_batch_size,
            learning_rate=self.experiment_config.training.learning_rate,
            weight_decay=self.experiment_config.training.weight_decay,
            num_train_epochs=self.experiment_config.training.num_train_epochs,
            warmup_ratio=self.experiment_config.training.warmup_ratio,
            gradient_accumulation_steps=self.experiment_config.training.gradient_accumulation_steps,
            logging_steps=self.experiment_config.training.logging_steps,
            save_steps=self.experiment_config.training.save_steps,
            fp16=self.experiment_config.training.fp16,
            bf16=self.experiment_config.training.bf16,
            evaluation_strategy="steps",
            save_strategy="steps",
            report_to=[],
            logging_dir=str(output_dir / "logs"),
            load_best_model_at_end=False,
        )
        data_collator = self._transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = self._transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._build_classification_metrics(id2label),
        )
        trainer.train()
        evaluation = trainer.evaluate()
        final_model_dir = ensure_directory(output_dir / "final_model")
        trainer.save_model(str(final_model_dir))
        history = [dict(row) for row in trainer.state.log_history]
        checkpoint_index_path = self._persist_checkpoints(
            output_dir,
            {
                "task_type": "sequence_classification",
                "label2id": label2id,
                "evaluation": evaluation,
            },
        )
        return TrainingArtifact(
            summary={
                "task_type": self.experiment_config.training.task_type,
                "evaluation": evaluation,
                "output_dir": str(output_dir),
                "final_model_dir": str(final_model_dir),
                "label2id": label2id,
            },
            history=history,
            checkpoint_index_path=checkpoint_index_path,
        )

    def _build_dataset_bundle(
        self,
        train_tasks: list[TaskSpec],
        eval_tasks: list[TaskSpec],
        include_labels: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "train": [self._build_example(task, include_labels=include_labels) for task in train_tasks],
            "eval": [self._build_example(task, include_labels=include_labels) for task in eval_tasks],
        }

    def _build_example(self, task: TaskSpec, include_labels: bool) -> dict[str, Any]:
        if task.group.value == "translation":
            input_text = (
                f"Translate from {task.metadata.get('source_language', 'source')} "
                f"to {task.metadata.get('target_language', 'target')}: "
                f"{task.metadata.get('source_text', task.description)}"
            )
        elif task.group.value == "software_engineering":
            failing_tests = "; ".join(task.metadata.get("failing_tests", []))
            regression_tests = "; ".join(task.metadata.get("regression_tests", []))
            input_text = (
                f"repo: {task.metadata.get('repo', '')} "
                f"issue: {task.metadata.get('issue_text', task.description)} "
                f"hints: {task.metadata.get('hints_text', '')} "
                f"failing_tests: {failing_tests} "
                f"regression_tests: {regression_tests}"
            )
        elif task.dataset_name.lower() == "squad":
            input_text = f"question: {task.description} context: {task.metadata.get('context', '')}"
        elif task.metadata.get("choices"):
            choices = " ".join(
                f"{choice['label']}: {choice['text']}" for choice in task.metadata.get("choices", [])
            )
            input_text = f"question: {task.description} context: {task.metadata.get('context', '')} choices: {choices}"
        else:
            input_text = f"question: {task.description} context: {task.metadata.get('context', '')}"
        example = {
            "task_id": task.task_id,
            "input_text": input_text.strip(),
            "target_text": str(task.metadata.get("reference_answer", task.metadata.get("reference_text", ""))).strip(),
        }
        if include_labels:
            example["label_text"] = example["target_text"]
        return example

    def _build_generative_metrics(self, tokenizer, eval_tasks: list[TaskSpec]):
        def compute_metrics(eval_prediction) -> dict[str, float]:
            predictions, labels = eval_prediction
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            cleaned_labels = []
            for row in labels:
                cleaned_labels.append([token if token >= 0 else tokenizer.pad_token_id for token in row])
            decoded_labels = tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)
            tasks = eval_tasks[: len(decoded_predictions)]
            results = [
                TaskExecutionResult(
                    success=True,
                    response=prediction,
                    used_skills=[],
                    reward=0.0,
                    resource_spent=0.0,
                )
                for prediction in decoded_predictions
            ]
            evaluation_tasks = [
                replace(
                    task,
                    metadata={
                        **task.metadata,
                        "reference_text": label,
                        "reference_answers": list(task.metadata.get("reference_answers", [])) or [label],
                    },
                )
                for task, label in zip(tasks, decoded_labels, strict=False)
            ]
            return self._evaluator.evaluate(evaluation_tasks, results, budget=1.0)

        return compute_metrics

    def _build_classification_metrics(self, id2label: dict[int, str]):
        def compute_metrics(eval_prediction) -> dict[str, float]:
            predictions, labels = eval_prediction
            if self._torch is None:
                predicted_ids = [int(max(range(len(row)), key=lambda index: row[index])) for row in predictions]
            else:
                tensor = self._torch.tensor(predictions)
                predicted_ids = tensor.argmax(dim=-1).tolist()
            predicted_labels = [id2label[int(index)] for index in predicted_ids]
            reference_labels = [id2label[int(index)] for index in labels.tolist()] if hasattr(labels, "tolist") else [id2label[int(index)] for index in labels]
            tasks = [
                TaskSpec(
                    task_id=f"eval-{index}",
                    title="classification",
                    description="classification",
                    group=self.experiment_config.task_group,
                    dataset_name=self.experiment_config.dataset,
                    capability_tags=[],
                    metadata={"reference_text": label, "reference_answers": [label]},
                )
                for index, label in enumerate(reference_labels)
            ]
            results = [
                TaskExecutionResult(success=True, response=label, used_skills=[], reward=0.0, resource_spent=0.0)
                for label in predicted_labels
            ]
            return self._evaluator.evaluate(tasks, results, budget=1.0)

        return compute_metrics

    def _persist_checkpoints(self, output_dir: Path, metadata: dict[str, Any]) -> str | None:
        checkpoints = list_checkpoint_directories(output_dir)
        if not self.experiment_config.persistence.save_checkpoints:
            return None
        return persist_checkpoint_index(
            output_dir,
            {
                **metadata,
                "checkpoints": checkpoints,
            },
        )
