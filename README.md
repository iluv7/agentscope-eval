# agentscope-eval

面向 **AgentScope 项目**的评测层：提交 AgentScope 运行记录，检查执行状态、工具调用和回答质量，返回分数、原因及批量汇总。底层使用 [DeepEval](https://github.com/confident-ai/deepeval) 4.1.8。

An evaluation layer for AgentScope, powered by DeepEval. Evaluate recorded AgentScope replies, tool execution, and answer quality through a local API or Python adapter.

这是独立项目，不代表 AgentScope 或 DeepEval 官方。

## 启动

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/iluv7/agentscope-eval.git
cd agentscope-eval
uv sync --locked
uv run agentscope-eval
```

打开 <http://127.0.0.1:8787/docs>，可以直接提交请求。修改端口使用 `uv run agentscope-eval --port 8788`。

依赖从 PyPI 安装，版本由 `uv.lock` 固定，无需额外克隆 DeepEval。评测服务读取 AgentScope 导出的 JSON；AgentScope 安装在被测项目的环境里，服务不需要再次安装或运行它。

## 评测 AgentScope 运行记录

专用入口接收 AgentScope **2.x** 的消息块格式（本地已对 2.0.7.post1 的原生 `Msg` 验证），不适用于 1.x 的 `tool_use` 格式。示例由真实 AgentScope 消息对象序列化生成，包含一次工具调用及最终回答：

```bash
curl -s http://127.0.0.1:8787/v1/agentscope/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/agentscope.json
```

默认示例评测 `execution_success`、`tool_correctness` 和 `exact_match`，均不需要模型密钥，预期汇总为：

```json
{"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 1.0}
```

`results` 中包含每项指标的 `score`、`threshold`、`status`、`reason` 和 `duration_ms`。

适配层保留调用顺序，按工具调用 ID 配对返回值，并提取最后一次工具调用/结果之后的文本作为最终回答。`thinking`、`hint` 和二进制块不会混入回答。未知块类型、重复调用 ID、无法解析的参数和不匹配的结果会返回 422。

### 从现有 AgentScope 事件流采集

在被测 AgentScope 项目现有的事件消费循环里，使用原生 `Msg.append_event()` 聚合本轮记录：

```python
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg

record = None
async for event in agent.reply_stream(user_msg):
    if isinstance(event, ReplyStartEvent) and record is None:
        record = Msg(
            id=event.reply_id, name=event.name, role="assistant", content=[]
        )
    elif record is not None and getattr(event, "reply_id", None) == record.id:
        record.append_event(event)

payload = {
    "cases": [
        {
            "case_id": "case-001",
            "input": user_msg.get_text_content(),
            "reply": record.model_dump(mode="json"),
        }
    ],
    "metrics": [{"name": "execution_success", "threshold": 1}],
}
```

将 `payload` POST 到 `/v1/agentscope/evaluate`。需要人工确认/外部执行的回复应先完成原有恢复流程，继续向同一条记录追加事件，直到收到 `ReplyEndEvent`。缺少结束原因的未完成记录会返回 422。

必须提交完整聚合记录：仅传 `agent.reply()` 的最终回答可能缺少工具调用与结果，无法可靠评估工具行为。当前按单次 reply 评测；多 Agent 分别按 `reply_id` 采集，压缩后的会话 context 不能替代原始记录。

`examples/tools.json` 和 `examples/answers.json` 则展示内部标准化接口 `/v1/evaluate`，适用于已经整理好评测字段的 AgentScope 测试代码。

## 配置 LLM 评委

```bash
cp .env.example .env
```

在 `.env` 设置 `EVAL_JUDGE_MODEL` 和 `EVAL_JUDGE_API_KEY`，必要时设置 `EVAL_JUDGE_BASE_URL`，然后重启服务。评委通过 Chat Completions 的 `response_format: json_object` 返回 JSON；需选用支持该接口和 JSON 模式的模型。本地模型服务若不校验 API key，可以填写占位值 `local`。

```bash
curl -s http://127.0.0.1:8787/v1/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/answers.json
```

评委收到的是待评分内容和评分指令。服务不会运行被测 Agent，也不会把 `expected_output` 当作待测 Agent 的输入。模型评委会调用你配置的外部或本地模型服务；密钥只在服务端配置。

## 指标语义

| 指标 | 必填补充数据 | 是否调用模型 | 评分含义 |
|---|---|---|---|
| `execution_success` | `finished_reason`、`tool_result_states`（专用入口自动提取） | 否 | 回复正常 completed 且全部工具 success 得 1，否则 0 |
| `exact_match` | `expected_output` | 否 | 字符串完全相同得 1，否则 0；保留空白差异 |
| `tool_correctness` | `tools_called`、`expected_tools` | 否 | DeepEval 精确比较工具数量、顺序、名称和输入参数；不比较返回内容 |
| `answer_correctness` | `expected_output` | 是 | GEval 按参考答案评估语义正确性 |
| `answer_relevancy` | 无 | 是 | DeepEval 评估回答是否回应问题 |
| `faithfulness` | 非空 `retrieval_context` | 是 | DeepEval 评估回答是否有检索资料支持 |

所有用例都必须提供 `case_id`、`input`、`actual_output`。同批次的所有指标应用于所有用例。`threshold` 默认 0.7，分数大于等于阈值即通过。无需调用工具时明确提交空数组 `[]`；省略字段表示没有采集数据，会被拒绝。

工具正确性是调用匹配，不证明工具执行成功，应结合 `execution_success` 使用。后者衡量运行是否正常结束，不等于业务任务完成度；例如成功调用创建工单工具，还应另行验证工单数据。当前版本没有暴露需要完整轨迹的 `TaskCompletionMetric`。

## 接口与错误

- `GET /health`：服务状态及评委是否已配置；不探测模型连通性。
- `GET /v1/metrics`：指标列表、输入要求和是否需要评委。
- `POST /v1/agentscope/evaluate`：读取 AgentScope 聚合消息并评分，最多 100 个用例、6 个不同指标。
- `POST /v1/evaluate`：使用已经标准化的评测输入评分。
- `GET /docs`：交互式 OpenAPI 文档。

缺少字段、重复 ID 或未知指标返回 422。未配置评委却请求 LLM 指标返回 503。评分期间的超时或模型错误保存在对应指标中（`status=error`，`score=null`），其余评分继续。HTTP 200 只表示已生成报告，需要进一步检查 `summary` 与逐项状态。

用例内任一指标 error，该用例状态为 error；否则任一指标不达标为 failed；全部达标为 passed。`pass_rate = passed / total`，错误用例也计入分母。所有指标独立保留结果，不计算意义不明确的跨指标平均分。

## Python 调用

```python
import asyncio
import json
from pathlib import Path

from agentscope_eval.agentscope import AgentScopeRequest
from agentscope_eval.config import Settings
from agentscope_eval.engine import Evaluator
from agentscope_eval.judge import JsonJudge


async def main():
    payload = json.loads(Path("examples/agentscope.json").read_text())
    request = AgentScopeRequest.model_validate(payload).to_evaluate_request()
    evaluator = Evaluator(Settings(), JsonJudge("", None))
    result = await evaluator.evaluate(request)
    print(result.model_dump_json(indent=2))


asyncio.run(main())
```

LLM 场景下创建一个 `AsyncOpenAI` 客户端传入 `JsonJudge`，完成后关闭客户端；参考 `src/agentscope_eval/api.py`。同一个事件循环可复用一个 Evaluator 来共享并发限制。

在已安装 AgentScope 的测试进程里，也可调用 `from agentscope_eval.agentscope import from_agentscope`，传入 `case_id`、`input` 和原生聚合 `Msg` 对象 `reply`，直接得到一个标准化 `EvalCase`。

## 实现与当前边界

```text
src/agentscope_eval/
├── api.py       HTTP 路由和客户端生命周期
├── agentscope.py AgentScope 2.x 消息到评测输入的转换
├── schemas.py   输入校验、输出契约
├── engine.py    独立指标实例、并发限制、错误隔离
├── judge.py     OpenAI 兼容 JSON 评委适配
├── config.py    服务端配置
└── cli.py       本地启动命令
```

直接调用新建指标的 `a_measure()`，不调用拥有全局运行状态和临时文件的 `deepeval.evaluate()`。默认关闭 DeepEval 遥测并使用只读文件模式；不配置 Confident AI 平台。活动指标数量按进程共享限制，默认 4；一个指标内部可能有多个模型请求，所以它不是精确的请求速率限制。默认每项指标执行超时 120 秒，不含排队时间。

这是无状态、同步等待报告的本地 API：不包含 Agent 执行、数据库、任务队列、历史报告或多租户平台。调用方负责保存返回的 JSON。评委成本和 Token 统计暂未提供；LLM 评分可能波动，比较版本时应固定模型和评分配置并人工抽查。服务默认仅监听 127.0.0.1，尚未实现面向公网的鉴权与流量控制。

## 开发验证

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

测试直接运行真实 DeepEval 工具匹配和 LLM 指标代码；LLM HTTP 返回由测试替身提供，不消耗线上模型费用。另覆盖缺参、错误隔离、超时、并发及 JSON 评委适配。

GitHub Actions 在 Python 3.11 和 3.13 上运行测试及 Ruff 检查，无需配置模型密钥。

## License

Copyright 2026 iluv7. Licensed under the [Apache License 2.0](LICENSE).

DeepEval 是独立安装的第三方依赖，遵循其自身的 [Apache-2.0 许可证](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md)。
