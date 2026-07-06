import os
import json


def save_state(current_path, state):
    """현재 state(messages, task_history)를 data/state.json으로 저장한다."""
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")

    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    # task_history는 supervisor 버전에서만 존재하므로 없으면 빈 리스트로 처리
    state_dict["task_history"] = [task.to_dict() for task in state.get("task_history", [])]

    with open(f"{current_path}/data/state.json", "w", encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)


def get_outline(current_path):
    """저장된 목차(outline.md)를 읽어온다. 없으면 안내 문구를 반환한다."""
    outline = '아직 작성된 목차가 없습니다.'

    if os.path.exists(f"{current_path}/data/outline.md"):
        with open(f"{current_path}/data/outline.md", "r", encoding='utf-8') as f:
            outline = f.read()
    return outline


def save_outline(current_path, outline):
    """작성된 목차(outline)를 data/outline.md로 저장한다."""
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")

    with open(f"{current_path}/data/outline.md", "w", encoding='utf-8') as f:
        f.write(outline)
    return outline
