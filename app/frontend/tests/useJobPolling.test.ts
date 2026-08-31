import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useJobPolling } from "../src/hooks/useJobPolling";
import * as client from "../src/api/client";
import type { JobStatus } from "../src/api/types";

function makeJob(overrides: Partial<JobStatus>): JobStatus {
  return {
    job_id: "job-1",
    job_type: "train_and_verify",
    status: "queued",
    auto_verify: true,
    created_at: "2026-08-30T12:00:00Z",
    log_tail: [],
    ...overrides,
  };
}

describe("useJobPolling", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("keeps polling while status is running, and stops exactly once it reaches done", async () => {
    const getJobSpy = vi
      .spyOn(client, "getJob")
      .mockResolvedValueOnce(makeJob({ status: "queued" }))
      .mockResolvedValueOnce(makeJob({ status: "running" }))
      .mockResolvedValueOnce(makeJob({ status: "done", revision: 1 }));

    const onSettled = vi.fn();
    const { result } = renderHook(() => useJobPolling("job-1", onSettled));

    await waitFor(() => expect(result.current.job?.status).toBe("queued"));
    expect(getJobSpy).toHaveBeenCalledTimes(1);

    await act(() => vi.advanceTimersByTimeAsync(1500));
    await waitFor(() => expect(result.current.job?.status).toBe("running"));
    expect(getJobSpy).toHaveBeenCalledTimes(2);

    await act(() => vi.advanceTimersByTimeAsync(1500));
    await waitFor(() => expect(result.current.job?.status).toBe("done"));
    expect(getJobSpy).toHaveBeenCalledTimes(3);
    expect(onSettled).toHaveBeenCalledTimes(1);
    expect(onSettled).toHaveBeenCalledWith(expect.objectContaining({ status: "done" }));

    // No further polling once settled -- advancing time again must not call getJob again.
    await act(() => vi.advanceTimersByTimeAsync(5000));
    expect(getJobSpy).toHaveBeenCalledTimes(3);
  });

  it("does a no-op when jobId is null", () => {
    const getJobSpy = vi.spyOn(client, "getJob");
    const { result } = renderHook(() => useJobPolling(null));
    expect(result.current.job).toBeNull();
    expect(getJobSpy).not.toHaveBeenCalled();
  });
});
