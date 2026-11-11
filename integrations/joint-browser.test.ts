/** No models, GPU, target generations or actual benchmark questions in these checks. */
import assert from "node:assert/strict";
import test from "node:test";
import { taskEvaluationSchema, taskPlan } from "../../src/benchmark/task-evaluation/protocol";
import { jointReport } from "../../src/benchmark/task-evaluation/joint-resource-report";

const sha = "a".repeat(64), base = `http://127.0.0.1:4173/models/${sha}/resolve/${sha}/`;
function fixture() { return {
  schema: "weight-space-browser-task-v2", study: "ifeval-browser-v1", jointMeasurement: true, jointPromptCount: 2,
  runId: "joint-fixture", configurationSha256: sha, arm: "baseline", sourceCheckpointSha256: sha, convertedArtifactSha256: sha,
  warmupPrompt: "Reply with OK.", totalLimitMs: 5400000,
  model: { id: `joint-${sha}`, modelUrl: base, modelLib: base + "model.wasm", quantization: "q4f16_1", requiredFeatures: ["shader-f16"], bufferSizeRequiredBytes: 1 },
  assets: ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json", "model.wasm"].map(n => ({ url: base + n, sha256: sha })),
  settings: { temperature: 0, topP: 1, seed: 42, maxTokens: 1024, contextWindow: 4096, warmups: 1, warmupTokens: 16, requestLimitMs: 120000, thinking: false },
  items: [{ item_id: "first", prompt: "Fixture one", rendered_input_tokens: 20 }, { item_id: "second", prompt: "Fixture two", rendered_input_tokens: 21 }],
}; }

test("joint mode retains one task plan rather than dispatching a repeat workload", () => {
  const value = taskEvaluationSchema.parse(fixture());
  const plan = taskPlan(value);
  assert.equal(plan.measuredRequests, 2);
  assert.equal(value.resourceComparison, undefined);
  assert.equal(plan.totalLimitMs, 5400000);
  assert.equal(plan.requestLimitMs, 120000);
});

test("joint schedule cannot mix resource workloads or silently widen count", () => {
  for (const mutate of [(x: any) => x.jointPromptCount = 3, (x: any) => x.resourceComparison = true,
    (x: any) => x.jointMeasurement = undefined, (x: any) => x.evaluationStage = "heldout",
    (x: any) => x.items[0].rendered_input_tokens = 4000]) {
    const value = fixture(); mutate(value);
    assert.equal(taskEvaluationSchema.safeParse(value).success, false);
  }
});

test("profile samples bind exact task IDs, clocks and warmup separately", () => {
  const value = taskEvaluationSchema.parse(fixture());
  const capture = jointReport(value);
  try {
    capture.observe("sample", { requestIndex: 0, warmup: true, outputTokens: 12, startedMs: 1, endedMs: 2 });
    capture.observe("sample", { requestIndex: 1, warmup: false, outputTokens: 25, startedMs: 3, endedMs: 8 });
    const report = capture.finish({ status: "interrupted", completedRequests: 1, unloadConfirmed: true }, false);
    assert.equal(report.samples[0].item_id, "__warmup__");
    assert.equal(report.samples[1].item_id, "first");
    assert.equal(report.samples[1].request_id, "generation-2");
    assert.equal(report.samples[1].startedMs, 3);
    assert.equal(report.artifact_sha256, sha);
    assert.equal(report.nativeTrace, undefined);
    assert.equal(report.profile.artifact, undefined);
  } finally { capture.finish({ status: "interrupted" }, true); }
});
