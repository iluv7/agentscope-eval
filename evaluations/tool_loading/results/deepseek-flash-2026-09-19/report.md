# AgentScope tool-loading benchmark

Evaluator version: 0.1.0. Runs: 12. Errors: 0.

Each row keeps provider, model, variant and settings separate. Rows marked `fixture` are synthetic checks, not model measurements.

| Group | Provider / model | Variant | Source | Scenario | Runs | First JSON | First schema | First arguments | First attempt | Eventual | Search R@K | Search P@K | MRR |
|---|---|---|---|---|---:|---|---|---|---|---|---:|---:|---:|
| 1 | deepseek / deepseek-flash | discovery | recorded | all | 12 | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 11/12 (91.7%) | 11/12 (91.7%) | 1.000 | 0.269 | 1.000 |
| 2 | deepseek / deepseek-flash | discovery | recorded | escaping_unicode | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1.000 | 0.200 | 1.000 |
| 3 | deepseek / deepseek-flash | discovery | recorded | json_as_text | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1.000 | 0.200 | 1.000 |
| 4 | deepseek / deepseek-flash | discovery | recorded | arrays_numbers | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 1.000 | 0.200 | 1.000 |
| 5 | deepseek / deepseek-flash | discovery | recorded | nested_null | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1.000 | 0.333 | 1.000 |
| 6 | deepseek / deepseek-flash | discovery | recorded | recursive_objects | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 1.000 | 0.350 | 1.000 |
| 7 | deepseek / deepseek-flash | discovery | recorded | escaping_nested | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 1.000 | 0.250 | 1.000 |
| 8 | deepseek / deepseek-flash | discovery | recorded | nested_arrays | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1.000 | 0.200 | 1.000 |
| 9 | deepseek / deepseek-flash | discovery | recorded | nested_arrays_null | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 1.000 | 0.225 | 1.000 |
| 10 | deepseek / deepseek-flash | discovery | recorded | objects_types | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) | 1.000 | 0.500 | 1.000 |

## Configuration details

Rates include not-reached stages in their eligible denominator. The JSON report also includes reached-only rates, stage counts, repairs, retries, labels, telemetry and the original evidence.

### Group 1

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 2

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 3

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 4

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 5

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 6

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 7

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 8

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 9

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```

### Group 10

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "variant": "discovery",
  "evaluation_scope": "trajectory",
  "schema_placement": "history",
  "agentscope_version": "2.0.8",
  "model_version": "DeepSeek-V4.1-Flash",
  "strict_output": false,
  "tool_choice": "auto",
  "temperature": 0.0,
  "seed": null,
  "catalog_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "prompt_version": "agentscope-tool-loading-full-trajectory:1.0.0",
  "source": "recorded"
}
```
