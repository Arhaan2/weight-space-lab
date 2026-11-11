/** BrowserLLM profiling over the actual scored IFEval requests, without a second workload. */
import { makeProfile, recordPoint, setSignal, startMainProfiler, clockMs } from "../metrics/browser-profiler";
import { taskPlan, type TaskEvaluation } from "./protocol";
import type { Phase } from "../schemas/profile";

export function jointReport(config: TaskEvaluation) {
  if (!config.jointMeasurement) throw Error("Joint measurement flag required");
  const profile = makeProfile(taskPlan(config), true);
  // makeProfile defaults to the upstream control model's catalog metadata.
  // This task is bound to the exact artifact hash below; never relabel that
  // unrelated control artifact as the model being measured here.
  delete (profile as any).artifact;
  const report: any = {
    schema: "weight-space-joint-browser-profile-v1", id: crypto.randomUUID(),
    workload: "ifeval", model: config.model.id, runtime: "WebLLM 0.2.86",
    modelLib: config.model.modelLib, quantization: config.model.quantization,
    configuration_sha256: config.configurationSha256,
    artifact_sha256: config.convertedArtifactSha256,
    environment: { userAgent: navigator.userAgent, platform: navigator.platform,
      hardwareConcurrency: navigator.hardwareConcurrency || 0 },
    profile, samples: [], status: "running", loadMs: null,
  };
  let phase: Phase = "baseline";
  const stop = startMainProfiler(profile, () => phase, () => {});
  return {
    report,
    observe(type: string, data: any) {
      if (type === "phase") phase = data;
      if (type === "event" && profile.events.length < 6000) profile.events.push(data);
      if (type === "point") recordPoint(profile, data);
      if (type === "signal") setSignal(profile, data);
      if (type === "network") profile.network = data;
      if (type === "active-start") profile.activeStartedMs = data;
      if (type === "active-end") profile.activeEndedMs = data;
      if (type === "count") profile.completedRequests = data;
      if (type === "loaded") report.loadMs = data;
      if (type === "sample") {
        const index = data.requestIndex;
        if (!Number.isInteger(index) || index < 0 || index > config.items.length ||
            data.warmup !== (index === 0)) throw Error("Joint sample identity differs");
        report.samples.push({ ...data, item_id: index === 0 ? "__warmup__" : config.items[index - 1].item_id,
          request_id: `generation-${index + 1}` });
      }
    },
    finish(done: any, forced: boolean) {
      stop();
      Object.assign(profile, { stopReason: done.reason,
        activeStartedMs: done.activeStartedMs ?? profile.activeStartedMs,
        activeEndedMs: done.activeEndedMs ?? profile.activeEndedMs,
        completedRequests: done.completedRequests ?? profile.completedRequests,
        unloadConfirmed: !forced && Boolean(done.unloadConfirmed), totalMs: clockMs() - profile.startedEpochMs });
      // These are task records, not the upstream fixed repeat-workload report schema.
      report.status = forced ? "failed" : done.status;
      profile.events.sort((a, b) => a.tMs - b.tMs);
      return report;
    },
  };
}
