import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getRevisions, isHeavyDepsMissing, postChat } from "../../src/api/client";

describe("api/client error handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("turns a 400 JSON error body into an ApiError with the backend's own detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "request does not resolve to anything genuinely unlearnable" }), {
          status: 400,
          statusText: "Bad Request",
        }),
      ),
    );

    await expect(postChat({ revision: 1, messages: [] })).rejects.toMatchObject({
      status: 400,
      message: "request does not resolve to anything genuinely unlearnable",
    });
  });

  it("turns a 503 into an ApiError that isHeavyDepsMissing() recognizes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "torch/transformers/peft are not installed" }), {
          status: 503,
          statusText: "Service Unavailable",
        }),
      ),
    );

    try {
      await postChat({ revision: 1, messages: [] });
      expect.fail("expected postChat to throw");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect(isHeavyDepsMissing(err)).toBe(true);
    }
  });

  it("does not treat a 400 as HeavyDepsMissing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "bad request" }), { status: 400 })),
    );
    try {
      await postChat({ revision: 1, messages: [] });
      expect.fail("expected postChat to throw");
    } catch (err) {
      expect(isHeavyDepsMissing(err)).toBe(false);
    }
  });

  it("wraps a network failure (fetch itself throwing) in a plain Error naming the backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(getRevisions()).rejects.toThrow(/Could not reach the ReTrace backend/);
  });

  it("parses a normal 200 JSON response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify([{ revision: 0 }]), { status: 200 })),
    );
    await expect(getRevisions()).resolves.toEqual([{ revision: 0 }]);
  });
});
