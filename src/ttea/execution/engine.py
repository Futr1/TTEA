from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..environments import EnvironmentAction, build_environment_adapter
from ..exceptions import EnvironmentIntegrationError
from ..runtime import TTEASystem
from ..types import DecisionType, ReasoningTrace, TaskExecutionResult, TaskGroup, TaskSpec
from ..utils import normalize_text


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class AssignmentContext:
    category_name: str
    leader_id: str
    primary_agent_id: str
    support_agent_ids: list[str]
    decision: DecisionType
    reasoning_trace: ReasoningTrace
    task: TaskSpec


@dataclass(slots=True)
class ExecutionArtifact:
    task: TaskSpec
    result: TaskExecutionResult
    benchmark: dict[str, Any]
    trace: dict[str, Any]


class TaskExecutionEngine:
    def __init__(self, system: TTEASystem) -> None:
        self.system = system
        self.environment = build_environment_adapter(system.experiment_config.environment)
        self.text_backend = system.reasoning.text_backend

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()

    def execute_tasks(self, tasks: list[TaskSpec]) -> list[ExecutionArtifact]:
        artifacts: list[ExecutionArtifact] = []
        try:
            for task in tasks:
                artifacts.append(self.execute_task(task))
            return artifacts
        finally:
            self.close()

    def execute_task(self, task: TaskSpec) -> ExecutionArtifact:
        contexts = self._prepare_assignment_contexts(task)
        if task.group == TaskGroup.WEB_NAVIGATION and self.environment is not None:
            try:
                artifact = self._execute_web_task(task, contexts)
            except EnvironmentIntegrationError as exc:
                return self._fallback_to_system(task, contexts, str(exc))
            except Exception as exc:
                return self._fallback_to_system(task, contexts, f"{type(exc).__name__}: {exc}")
        else:
            artifact = self._execute_text_task(task, contexts)
        self._record_system_effects(task, artifact.result, contexts)
        return artifact

    def _prepare_assignment_contexts(self, task: TaskSpec) -> list[AssignmentContext]:
        if self.system.module_switches["communication"]:
            self.system.synergy.synchronize(
                self.system.topology.global_leader,
                self.system.topology.category_leaders,
                self.system.global_memory,
            )
        plan = self.system.dispatcher.plan(
            self.system.topology,
            task,
            max_depth=self.system.experiment_config.runtime.decomposition_depth,
        )
        prepared: list[dict[str, Any]] = []
        observations = []
        agent_signatures: list[str] = []
        for assignment in plan.assignments:
            category_name = str(
                assignment.task.metadata.get(
                    "assigned_category",
                    next(iter(self.system.topology.category_leaders)),
                )
            )
            leader = self.system.topology.category_leaders[category_name]
            primary = next(member for member in leader.members if member.agent_id == assignment.primary_agent_id)
            reasoning_trace = self.system.reasoning.prepare(primary.agent_id, primary.role.value, assignment.task, self.system.state)
            observation = primary.build_observation(assignment.task, self.system.state)
            observations.append(observation)
            agent_signatures.append(primary.agent_id)
            skill_match = primary.skill_match(assignment.task.capability_tags)
            collaboration_need = min(
                1.0,
                (assignment.task.complexity + self.system.state.resources.blocked_ratio + reasoning_trace.confidence_bias) / 2.0,
            )
            if self.system.module_switches["top_level_objective"]:
                decision = self.system.assessment.evaluate(self.system.state, assignment.task, skill_match, collaboration_need).best
            else:
                decision = DecisionType.EXECUTE
            support_ids: list[str] = []
            if decision == DecisionType.REJECT and assignment.task.priority >= 0.70:
                decision = DecisionType.ASSIST
            if self.system.module_switches["evolution"] and decision == DecisionType.LEARN:
                self.system.evolution.learning.apply(primary, assignment.task.capability_tags)
            if decision == DecisionType.ASSIST or skill_match < 0.45:
                support_ids = self.system.dispatcher.request_assistance(self.system.topology, assignment)
            prepared.append(
                {
                    "category_name": category_name,
                    "leader": leader,
                    "primary": primary,
                    "support_ids": support_ids,
                    "decision": decision,
                    "reasoning_trace": reasoning_trace,
                    "task": assignment.task,
                }
            )
        batch_encoding = None
        if self.system.module_switches["communication"] and observations:
            batch_encoding = self.system.encoder.encode_batch(observations, agent_signatures=agent_signatures)

        contexts: list[AssignmentContext] = []
        for index, item in enumerate(prepared):
            if self.system.module_switches["communication"] and batch_encoding is not None:
                encoded = batch_encoding.vectors[index]
                macro = self.system.macro_adapter.apply(encoded)
                micro = self.system.micro_adapter.apply(macro, item["primary"].agent_id)
                message = self.system.bridge.encode(micro)
                diagnostics = dict(batch_encoding.diagnostics)
                item["reasoning_trace"].metadata["communication"] = diagnostics
                item["leader"].capability_map.record_observation(f"{item['primary'].agent_id}: {message}")
                self.system.global_memory.record_message(f"{item['primary'].agent_id}->{item['leader'].agent_id}: {message}")
                self.system.global_memory.record_message(
                    f"fusion_mode={diagnostics.get('fusion_mode')} communication_rate={diagnostics.get('communication_rate', 0.0):.4f}"
                )
                self.system.global_memory.record_message(
                    f"prompt_backend={item['reasoning_trace'].metadata.get('prompt_backend')} tokenizer={item['reasoning_trace'].metadata.get('tokenizer_backend')}"
                )
            contexts.append(
                AssignmentContext(
                    category_name=item["category_name"],
                    leader_id=item["leader"].agent_id,
                    primary_agent_id=item["primary"].agent_id,
                    support_agent_ids=item["support_ids"],
                    decision=item["decision"],
                    reasoning_trace=item["reasoning_trace"],
                    task=item["task"],
                )
            )
        return contexts

    def _execute_text_task(self, task: TaskSpec, contexts: list[AssignmentContext]) -> ExecutionArtifact:
        prompt = self._build_text_prompt(task, contexts)
        generation = self.text_backend.generate(
            prompt,
            max_new_tokens=self._resolve_generation_limit(task),
            stop_strings=["\nObservation:", "\nAction:"],
        )
        response = self._coerce_text_response(task, generation.text)
        success = bool(response.strip())
        quality = 0.75 if generation.backend == "transformers" else 0.55
        if not success:
            quality = 0.0
        benchmark = {"backend": generation.backend, "model_family": generation.model_family}
        result = TaskExecutionResult(
            success=success,
            response=response,
            used_skills=list({tag for context in contexts for tag in context.task.capability_tags}) or list(task.capability_tags),
            reward=max(0.1, task.priority) if success else -0.1,
            resource_spent=min(
                self.system.experiment_config.runtime.resource_budget,
                0.20 + generation.generated_tokens / 128.0 + task.complexity * 0.35,
            ),
            evidence=[f"{context.primary_agent_id}:{context.category_name}" for context in contexts],
            metrics={"quality": quality},
            metadata={
                "backend": generation.backend,
                "model_family": generation.model_family,
                "generated_tokens": generation.generated_tokens,
            },
        )
        trace = {
            "mode": "text_generation",
            "prompt": prompt,
            "generation": {
                "backend": generation.backend,
                "model_family": generation.model_family,
                "generated_tokens": generation.generated_tokens,
                "metadata": generation.metadata,
            },
            "assignments": [self._serialize_assignment(context) for context in contexts],
        }
        return ExecutionArtifact(task=task, result=result, benchmark=benchmark, trace=trace)

    def _execute_web_task(self, task: TaskSpec, contexts: list[AssignmentContext]) -> ExecutionArtifact:
        observation = self.environment.reset(task)
        initial_observation = observation.to_dict()
        max_steps = min(task.metadata.get("max_steps", self.system.experiment_config.runtime.max_steps), self.system.experiment_config.runtime.max_steps)
        trajectory: list[EnvironmentAction] = []
        step_traces: list[dict[str, Any]] = []
        benchmark: dict[str, Any] = {"backend": "environment"}
        for index in range(int(max_steps)):
            context = contexts[index % max(1, len(contexts))]
            prompt = self._build_web_prompt(task, context, observation, trajectory, step_index=index)
            generation = self.text_backend.generate(prompt, max_new_tokens=128, stop_strings=["\nObservation:", "\nResult:"])
            action = self._parse_environment_action(generation.text, task, observation, index)
            if action.action_type == "stop":
                break
            trajectory.append(action)
            step = self.environment.step(action)
            observation = step.observation
            intermediate = self.environment.evaluate(task, trajectory)
            step_traces.append(
                {
                    "step_index": index,
                    "agent_id": context.primary_agent_id,
                    "prompt": prompt,
                    "model_output": generation.text,
                    "action": action.to_dict(),
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "benchmark": intermediate,
                }
            )
            benchmark = intermediate
            if step.terminated or step.truncated or intermediate.get("benchmark_success"):
                break
        success = bool(benchmark.get("benchmark_success") or benchmark.get("success"))
        result = TaskExecutionResult(
            success=success,
            response=json.dumps(
                {
                    "goal": task.title,
                    "status": "completed" if success else "blocked",
                    "steps": len(trajectory),
                    "final_url": observation.url,
                },
                ensure_ascii=False,
            ),
            used_skills=list({tag for context in contexts for tag in context.task.capability_tags}) or list(task.capability_tags),
            reward=1.0 if success else -0.2,
            resource_spent=min(
                self.system.experiment_config.runtime.resource_budget,
                0.25 + len(trajectory) * 0.08 + task.complexity * 0.45,
            ),
            evidence=[json.dumps(action.to_dict(), ensure_ascii=False) for action in trajectory],
            metrics={"quality": 1.0 if success else 0.0},
            metadata={
                "trajectory_length": len(trajectory),
                "final_url": observation.url,
                "environment_provider": self.system.experiment_config.environment.provider,
            },
        )
        trace = {
            "mode": "web_environment",
            "initial_observation": initial_observation,
            "final_observation": observation.to_dict(),
            "steps": step_traces,
            "assignments": [self._serialize_assignment(context) for context in contexts],
        }
        return ExecutionArtifact(task=task, result=result, benchmark=benchmark, trace=trace)

    def _fallback_to_system(self, task: TaskSpec, contexts: list[AssignmentContext], reason: str) -> ExecutionArtifact:
        result = self.system.run_task(task)
        result.metadata["execution_mode"] = "system_fallback"
        result.metadata["fallback_reason"] = reason
        trace = {
            "mode": "system_fallback",
            "reason": reason,
            "assignments": [self._serialize_assignment(context) for context in contexts],
        }
        benchmark = {"backend": "system_fallback", "reason": reason, "benchmark_success": result.success}
        return ExecutionArtifact(task=task, result=result, benchmark=benchmark, trace=trace)

    def _build_text_prompt(self, task: TaskSpec, contexts: list[AssignmentContext]) -> str:
        assignment_block = "\n".join(
            f"- agent={context.primary_agent_id} category={context.category_name} decision={context.decision.value}"
            for context in contexts
        )
        if task.group == TaskGroup.TRANSLATION:
            return (
                "You are operating inside a TTEA system.\n"
                f"Task: Translate the source text from {task.metadata.get('source_language', 'source')} "
                f"to {task.metadata.get('target_language', 'target')}.\n"
                "Return only the translation.\n"
                f"Assignments:\n{assignment_block}\n"
                f"Source:\n{task.metadata.get('source_text', task.description)}"
            )
        if task.group == TaskGroup.KNOWLEDGE_ENHANCEMENT:
            choices = task.metadata.get("choices", [])
            choice_block = ""
            if choices:
                choice_block = "\nChoices:\n" + "\n".join(f"- {item['label']}: {item['text']}" for item in choices)
            return (
                "You are operating inside a TTEA system.\n"
                "Answer the question using the provided context. "
                "If choices are given, return the best matching label or option text.\n"
                f"Assignments:\n{assignment_block}\n"
                f"Question:\n{task.description}\n"
                f"Context:\n{task.metadata.get('context', '')}{choice_block}\n"
                "Answer:"
            )
        return (
            "You are operating inside a TTEA system.\n"
            f"Assignments:\n{assignment_block}\n"
            f"Task:\n{task.description}"
        )

    def _build_web_prompt(
        self,
        task: TaskSpec,
        context: AssignmentContext,
        observation,
        trajectory: list[EnvironmentAction],
        step_index: int,
    ) -> str:
        history = "\n".join(
            f"{index + 1}. {action.action_type} selector={action.selector or '-'} text={action.text or action.value or '-'}"
            for index, action in enumerate(trajectory[-8:])
        )
        return (
            "You are controlling a live browser environment through structured actions.\n"
            "Return exactly one JSON object with keys: action_type, selector, text, value, url, key, metadata.\n"
            "Allowed action_type values: goto, click, type, press, select, check, uncheck, wait, stop.\n"
            f"Agent: {context.primary_agent_id} ({context.category_name})\n"
            f"Decision: {context.decision.value}\n"
            f"Task goal: {task.title}\n"
            f"Task description: {task.description}\n"
            f"Step: {step_index}\n"
            f"Current URL: {observation.url}\n"
            f"Page title: {observation.title}\n"
            f"Page content:\n{observation.content}\n"
            f"Action history:\n{history or 'none'}\n"
            f"Action hints: {task.metadata.get('action_hints', [])}\n"
            "JSON:"
        )

    def _parse_environment_action(
        self,
        generated_text: str,
        task: TaskSpec,
        observation,
        step_index: int,
    ) -> EnvironmentAction:
        payload: dict[str, Any] | None = None
        match = _JSON_OBJECT_PATTERN.search(generated_text)
        if match is not None:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            payload = self._heuristic_action(generated_text, task, observation, step_index)
        action_type = str(payload.get("action_type", "wait")).strip().lower()
        return EnvironmentAction(
            action_type=action_type,
            selector=str(payload.get("selector", "")),
            text=str(payload.get("text", "")),
            value=str(payload.get("value", "")),
            url=str(payload.get("url", "")),
            key=str(payload.get("key", "")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    def _heuristic_action(
        self,
        generated_text: str,
        task: TaskSpec,
        observation,
        step_index: int,
    ) -> dict[str, Any]:
        lowered = generated_text.lower()
        if step_index == 0 and task.metadata.get("start_url") and observation.url != task.metadata.get("start_url"):
            return {"action_type": "goto", "url": str(task.metadata.get("start_url"))}
        if "click" in lowered:
            return {"action_type": "click", "selector": self._guess_selector(generated_text)}
        if "type" in lowered or "input" in lowered:
            return {
                "action_type": "type",
                "selector": self._guess_selector(generated_text),
                "text": task.description,
            }
        if "press" in lowered or "enter" in lowered:
            return {"action_type": "press", "selector": self._guess_selector(generated_text), "key": "Enter"}
        if "stop" in lowered or "done" in lowered:
            return {"action_type": "stop"}
        return {"action_type": "wait", "metadata": {"seconds": 1.0}}

    def _guess_selector(self, text: str) -> str:
        match = re.search(r"(#[-_a-zA-Z0-9]+|\.[-_a-zA-Z0-9]+)", text)
        if match is not None:
            return match.group(1)
        quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
        if quoted is not None:
            content = quoted.group(1)
            return f"text={content}"
        return "body"

    def _coerce_text_response(self, task: TaskSpec, generated_text: str) -> str:
        text = generated_text.strip()
        if task.group == TaskGroup.TRANSLATION:
            return text
        choices = task.metadata.get("choices", [])
        if task.dataset_name.lower() in {"pubhealth", "arc-challenge"} and choices:
            return self._closest_choice(text, choices)
        if task.dataset_name.lower() == "squad":
            return text.split("\n", 1)[0].strip()
        return text

    def _closest_choice(self, prediction: str, choices: list[dict[str, str]]) -> str:
        normalized_prediction = normalize_text(prediction)
        best_choice = choices[0]["text"]
        best_score = -1.0
        for choice in choices:
            label = normalize_text(choice["label"])
            text = normalize_text(choice["text"])
            score = 0.0
            if normalized_prediction == label or normalized_prediction == text:
                score += 10.0
            if label and label in normalized_prediction:
                score += 3.0
            if text and text in normalized_prediction:
                score += 5.0
            overlap = set(normalized_prediction.split()) & set(text.split())
            score += len(overlap)
            if score > best_score:
                best_score = score
                best_choice = choice["text"]
        return best_choice

    def _resolve_generation_limit(self, task: TaskSpec) -> int:
        if task.group == TaskGroup.TRANSLATION:
            return 192
        if task.dataset_name.lower() == "asqa":
            return 256
        return self.system.platform_config.models.generation.get("max_new_tokens", 96)

    def _record_system_effects(
        self,
        task: TaskSpec,
        result: TaskExecutionResult,
        contexts: list[AssignmentContext],
    ) -> None:
        self.system._update_state(task, result)
        if self.system.module_switches["evolution"] and contexts:
            primary_context = contexts[0]
            leader = self.system.topology.category_leaders[primary_context.category_name]
            primary = next(member for member in leader.members if member.agent_id == primary_context.primary_agent_id)
            primary.touch_skills(result.used_skills, self.system.state.step)
            self.system.evolution.skill_reinforcement.apply(
                primary,
                result.used_skills,
                local_reward=result.reward,
                delta_utility=result.metrics.get("quality", 0.0),
                current_step=self.system.state.step,
            )
            leader.refresh_capability_map()
            self.system.maintenance_log.append(self.system.evolution.maintain(self.system.topology.category_leaders, self.system.state.step))
        self.system.state = self.system.state.__class__(
            step=self.system.state.step + 1,
            utility=self.system.state.utility,
            resources=self.system.state.resources,
            completed_tasks=self.system.state.completed_tasks,
            failed_tasks=self.system.state.failed_tasks,
        )

    def _serialize_assignment(self, context: AssignmentContext) -> dict[str, Any]:
        return {
            "leader_id": context.leader_id,
            "category_name": context.category_name,
            "primary_agent_id": context.primary_agent_id,
            "support_agent_ids": list(context.support_agent_ids),
            "decision": context.decision.value,
            "task_id": context.task.task_id,
            "reasoning_trace": {
                "token_count": context.reasoning_trace.token_count,
                "keywords": list(context.reasoning_trace.keywords),
                "metadata": dict(context.reasoning_trace.metadata),
            },
        }
