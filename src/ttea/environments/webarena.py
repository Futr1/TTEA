from __future__ import annotations

import json
import time
from typing import Any

from ..config import EnvironmentConfig
from ..exceptions import EnvironmentIntegrationError
from ..integrations import import_playwright_sync
from ..types import TaskSpec
from .base import EnvironmentAction, EnvironmentObservation, EnvironmentStep, WebEnvironmentAdapter


class WebArenaEnvironmentAdapter(WebEnvironmentAdapter):
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self._playwright_module = import_playwright_sync()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_session(self) -> None:
        if self._page is not None:
            return
        if self._playwright_module is None:
            raise EnvironmentIntegrationError("playwright.sync_api is not installed.")
        try:
            self._playwright = self._playwright_module.sync_playwright().start()
            browser_launcher = getattr(self._playwright, self.config.browser_name)
            self._browser = browser_launcher.launch(headless=self.config.headless)
            self._context = self._browser.new_context(
                viewport={"width": self.config.viewport_width, "height": self.config.viewport_height}
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.config.task_timeout_ms)
        except Exception as exc:
            raise EnvironmentIntegrationError(f"Failed to initialize WebArena browser context: {exc}") from exc

    def reset(self, task: TaskSpec) -> EnvironmentObservation:
        self._ensure_session()
        target_url = str(task.metadata.get("start_url") or task.metadata.get("url") or self.config.base_url)
        if not target_url:
            raise EnvironmentIntegrationError("WebArena task does not define a start URL and no base_url is configured.")
        self._page.goto(target_url, wait_until="domcontentloaded")
        time.sleep(self.config.action_delay_ms / 1000.0)
        return self._observation()

    def step(self, action: EnvironmentAction) -> EnvironmentStep:
        self._ensure_session()
        if self._page is None:
            raise EnvironmentIntegrationError("WebArena page is not initialized.")
        try:
            self._apply_action(action)
            time.sleep(self.config.action_delay_ms / 1000.0)
            observation = self._observation()
            return EnvironmentStep(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"action_type": action.action_type},
            )
        except Exception as exc:
            observation = self._observation()
            return EnvironmentStep(
                observation=observation,
                reward=-0.1,
                terminated=False,
                truncated=False,
                info={"action_type": action.action_type, "error": str(exc)},
            )

    def _apply_action(self, action: EnvironmentAction) -> None:
        if action.action_type == "goto":
            self._page.goto(action.url or self.config.base_url, wait_until="domcontentloaded")
            return
        if action.action_type == "click":
            self._page.locator(action.selector).click()
            return
        if action.action_type == "type":
            self._page.locator(action.selector).fill(action.text or action.value)
            return
        if action.action_type == "press":
            self._page.locator(action.selector).press(action.key or "Enter")
            return
        if action.action_type == "select":
            self._page.locator(action.selector).select_option(action.value)
            return
        if action.action_type == "check":
            self._page.locator(action.selector).check()
            return
        if action.action_type == "uncheck":
            self._page.locator(action.selector).uncheck()
            return
        if action.action_type == "wait":
            time.sleep(float(action.metadata.get("seconds", 1.0)))
            return
        raise EnvironmentIntegrationError(f"Unsupported WebArena action type: {action.action_type}")

    def _observation(self) -> EnvironmentObservation:
        if self._page is None:
            raise EnvironmentIntegrationError("WebArena page is not initialized.")
        content = self._page.locator("body").inner_text(timeout=self.config.task_timeout_ms)
        return EnvironmentObservation(
            url=self._page.url,
            title=self._page.title(),
            content=content[: self.config.observation_max_chars],
            metadata={"html_length": len(content)},
        )

    def evaluate(self, task: TaskSpec, trajectory: list[EnvironmentAction]) -> dict[str, Any]:
        self._ensure_session()
        if self._page is None:
            raise EnvironmentIntegrationError("WebArena page is not initialized.")
        current_url = self._page.url
        page_text = self._page.locator("body").inner_text(timeout=self.config.task_timeout_ms)
        rules = task.metadata.get("evaluation", {})
        required_url = str(rules.get("expected_url_contains", task.metadata.get("expected_url_contains", "")))
        required_text = str(rules.get("required_text", task.metadata.get("required_text", "")))
        success_selectors = list(rules.get("success_selectors", task.metadata.get("success_selectors", [])))

        url_ok = True if not required_url else required_url in current_url
        text_ok = True if not required_text else required_text.lower() in page_text.lower()
        selector_ok = True
        selector_hits: list[str] = []
        for selector in success_selectors:
            try:
                count = self._page.locator(selector).count()
                if count <= 0:
                    selector_ok = False
                else:
                    selector_hits.append(selector)
            except Exception:
                selector_ok = False

        success = url_ok and text_ok and selector_ok
        return {
            "success": success,
            "benchmark_success": success,
            "current_url": current_url,
            "required_url": required_url,
            "required_text": required_text,
            "selector_hits": selector_hits,
            "trajectory": [json.dumps(action.to_dict(), ensure_ascii=False) for action in trajectory],
        }

    def close(self) -> None:
        if self._page is not None:
            self._page.close()
            self._page = None
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
