from __future__ import annotations

import json
import os
import time

from fastapi.testclient import TestClient

# Smoke tests exercise the non-model parts without requiring a downloaded
# Ollama model.  Normal application startup keeps the strict model gate on.
os.environ.setdefault("RESUME_AGENT_STRICT_MODEL_GATE", "false")

from app.core.utils import sha256_text
from app.main import app


def main() -> None:
    with TestClient(app) as client:
        token = app.state.resume.settings.internal_token
        headers = {"X-Resume-Agent-Token": token}
        assert client.get("/api/environment", headers=headers).status_code == 200
        response = client.post(
            "/api/jobs",
            headers=headers,
            json={
                "title": "Python 后端工程师",
                "city": "深圳",
                "skills": ["Python", "FastAPI", "LangGraph"],
                "responsibilities": ["建设知识库问答系统"],
                "requirements": ["熟悉 PostgreSQL"],
            },
        )
        response.raise_for_status()
        task_response = client.post(
            "/api/tasks",
            headers=headers,
            json={
                "job_id": response.json()["job_id"],
                "candidate_count": 3,
                "base_facts": {
                    "name": "测试用户",
                    "phone": "13800000000",
                    "email": "test@example.com",
                    "objective": "Python 后端工程师",
                },
            },
        )
        task_response.raise_for_status()
        task_id = task_response.json()["task_id"]
        for _ in range(30):
            time.sleep(0.1)
            task = client.get(f"/api/tasks/{task_id}", headers=headers).json()
            if task["status"] not in {"queued", "running"}:
                break
        assert task["status"] == "waiting_user", task
        selected_candidate = next(
            item for item in task["candidates"] if item.get("status") != "duplicate"
        )
        action = {
            "thread_id": task["thread_id"],
            "checkpoint_version": task["checkpoint_version"],
            "action": "select",
            "payload": {"candidate_id": selected_candidate["candidate_id"]},
        }
        selected = client.post(f"/api/tasks/{task_id}/resume", headers=headers, json=action)
        assert selected.status_code == 200, selected.text
        selected_task = selected.json()
        assert selected_task["status"] == "waiting_user", selected_task
        assert selected_task["current_node"] == "candidate_confirmation", selected_task
        assert selected_task["selected_candidate_id"] == selected_candidate["candidate_id"], selected_task

        # Selection and formal confirmation are intentionally separate.  The
        # latter must bind every candidate field to the exact value hash that
        # the user reviewed before an immutable resume snapshot is created.
        candidate = next(
            item for item in selected_task["candidates"] if item["candidate_id"] == selected_task["selected_candidate_id"]
        )
        confirmations = []
        for field in ("title", "period", "introduction", "tech_stack", "solutions", "results"):
            value = candidate[field]
            serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            confirmations.append({"field_id": field, "value_hash": sha256_text(serialized), "confirmed": True})
        confirmed = client.post(
            f"/api/tasks/{task_id}/confirmations",
            headers=headers,
            json={
                "thread_id": selected_task["thread_id"],
                "checkpoint_version": selected_task["checkpoint_version"],
                "candidate_id": candidate["candidate_id"],
                "confirm": True,
                "confirmations": confirmations,
            },
        )
        confirmed.raise_for_status()
        project_step = confirmed.json()
        assert project_step["current_node"] == "project_application", project_step
        applied = client.post(
            f"/api/tasks/{task_id}/resume",
            headers=headers,
            json={
                "thread_id": project_step["thread_id"],
                "checkpoint_version": project_step["checkpoint_version"],
                "action": "approve",
                "payload": {"project_action": "copy_only"},
            },
        )
        applied.raise_for_status()
        layout_step = applied.json()
        assert layout_step["current_node"] == "layout_review", layout_step
        layout = client.post(
            f"/api/tasks/{task_id}/resume",
            headers=headers,
            json={
                "thread_id": layout_step["thread_id"],
                "checkpoint_version": layout_step["checkpoint_version"],
                "action": "approve",
                "payload": {"page_count": 1},
            },
        )
        layout.raise_for_status()
        preview_step = layout.json()
        assert preview_step["current_node"] == "preview_review", preview_step
        preview = client.post(
            f"/api/tasks/{task_id}/export-preview",
            headers=headers,
            json={
                "thread_id": preview_step["thread_id"],
                "checkpoint_version": preview_step["checkpoint_version"],
                "snapshot_id": preview_step["resume_snapshot_id"],
                "format": "markdown",
            },
        )
        preview.raise_for_status()
        preview_content = preview.json()["content"]
        assert candidate["title"] in preview_content, preview.json()
        unselected = next(
            (
                item
                for item in task["candidates"]
                if item["candidate_id"] != candidate["candidate_id"]
                and item["title"] != candidate["title"]
            ),
            None,
        )
        if unselected is not None:
            assert unselected["title"] not in preview_content, preview.json()
        final = client.post(
            f"/api/tasks/{task_id}/resume",
            headers=headers,
            json={
                "thread_id": preview_step["thread_id"],
                "checkpoint_version": preview_step["checkpoint_version"],
                "action": "confirm",
                "payload": {},
            },
        )
        final.raise_for_status()
        completed = final.json()
        assert completed["status"] == "completed", completed
        stale = client.post(f"/api/tasks/{task_id}/resume", headers=headers, json=action)
        assert stale.status_code == 409
    print("smoke ok")


if __name__ == "__main__":
    main()
