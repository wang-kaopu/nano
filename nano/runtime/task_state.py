"""一次 agent 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
TaskStatus = Literal["running", "completed", "stopped", "failed"]

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_TURN_LIMIT_REACHED = "turn_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_DELEGATE_FAILED = "delegate_failed"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"
STOP_REASON_USER_INTERRUPTED = "user_interrupted"
STOP_REASON_INVALID_TOOL_CALL_LIMIT_REACHED = "invalid_tool_call_limit_reached"


class TaskState(BaseModel):
    """描述一次 agent 运行的可持久化状态。"""

    model_config = ConfigDict(validate_assignment=True)

    run_id: str
    task_id: str
    user_request: str
    status: TaskStatus = STATUS_RUNNING
    tool_steps: int = Field(default=0, ge=0)
    invalid_tool_calls: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""

    @classmethod
    def create(cls, task_id: str, user_request: str, run_id: str = "") -> "TaskState":
        """创建一个新的运行状态并生成运行 ID。"""
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        """从结构化数据校验并恢复运行状态。"""
        return cls.model_validate(data)

    def record_attempt(self) -> "TaskState":
        """记录一次模型调用尝试。"""
        # attempt 统计的是“模型被调用了几轮”，不等于 tool_steps。
        self.attempts += 1
        return self

    def record_tool(self, name: str) -> "TaskState":
        """记录一次实际执行的工具调用。"""
        # tool_steps 只统计真正进入执行阶段的工具调用次数。
        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def record_invalid_tool(self, name: str) -> "TaskState":
        """记录一次被护栏拒绝的工具调用，不消耗正常工作步骤预算。"""
        self.invalid_tool_calls += 1
        self.last_tool = str(name or "")
        return self

    def stop(self, stop_reason: str, status: TaskStatus = STATUS_STOPPED, final_answer: str = "") -> "TaskState":
        """以指定停止原因结束当前运行。"""
        # stop_reason 和 status 分开存，是为了区分“怎么停的”和“停下时是什么状态”。
        self.status = status
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer: str = "") -> "TaskState":
        """标记运行达到最大步数。"""
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_turn_limit(self, final_answer: str = "") -> "TaskState":
        """标记运行达到最大模型循环次数。"""
        return self.stop(STOP_REASON_TURN_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer: str = "") -> "TaskState":
        """标记运行达到模型重试上限。"""
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_approval_denied(self, final_answer: str = "") -> "TaskState":
        """标记危险操作因用户拒绝审批而终止。"""
        return self.stop(STOP_REASON_APPROVAL_DENIED, final_answer=final_answer)

    def stop_model_error(self, final_answer: str = "") -> "TaskState":
        """标记模型调用失败。"""
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def stop_user_interrupted(self, final_answer: str = "") -> "TaskState":
        """标记当前请求由用户主动打断。"""
        return self.stop(STOP_REASON_USER_INTERRUPTED, final_answer=final_answer)

    def stop_invalid_tool_call_limit(self, final_answer: str = "") -> "TaskState":
        """标记运行因连续无效工具调用过多而结束。"""
        return self.stop(STOP_REASON_INVALID_TOOL_CALL_LIMIT_REACHED, final_answer=final_answer)

    def finish_success(self, final_answer: str) -> "TaskState":
        """标记运行成功并保存最终答案。"""
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
        return self

    def to_dict(self) -> dict[str, Any]:
        """返回用于 JSON 持久化的状态字典。"""
        return self.model_dump(mode="json")
