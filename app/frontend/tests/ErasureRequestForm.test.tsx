import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErasureRequestForm } from "../src/components/erasure/ErasureRequestForm";
import { HeavyDepsProvider } from "../src/components/layout/HeavyDepsContext";
import * as client from "../src/api/client";
import type { EntityListItem, ExampleRequest, JobStatus } from "../src/api/types";

const ENTITIES: EntityListItem[] = [
  { entity: "Silvergate Aerospace", entity_type: "company", fact_group_id: "G014" },
];
const EXAMPLES: ExampleRequest[] = [
  { name: "erase_silvergate_aerospace", entity: "Silvergate Aerospace", attribute: null, comment: "demo request" },
];

function renderForm() {
  return render(
    <HeavyDepsProvider>
      <ErasureRequestForm />
    </HeavyDepsProvider>,
  );
}

describe("ErasureRequestForm", () => {
  afterEach(() => vi.restoreAllMocks());

  it("disables submit until an entity or attribute is chosen, and enables it once one is", async () => {
    vi.spyOn(client, "getEntities").mockResolvedValue(ENTITIES);
    vi.spyOn(client, "getAttributes").mockResolvedValue({ company: ["ceo"], person: [] });
    vi.spyOn(client, "getExampleRequests").mockResolvedValue([]);
    vi.spyOn(client, "listJobs").mockResolvedValue([]);

    const user = userEvent.setup();
    renderForm();

    const submit = await screen.findByRole("button", { name: /submit erasure request/i });
    expect(submit).toBeDisabled();

    const entitySelect = screen.getByLabelText("Entity");
    await user.selectOptions(entitySelect, "Silvergate Aerospace");

    await waitFor(() => expect(submit).toBeEnabled());
  });

  it("pre-fills the form when an example request is picked", async () => {
    vi.spyOn(client, "getEntities").mockResolvedValue(ENTITIES);
    vi.spyOn(client, "getAttributes").mockResolvedValue({ company: ["ceo"], person: [] });
    vi.spyOn(client, "getExampleRequests").mockResolvedValue(EXAMPLES);
    vi.spyOn(client, "listJobs").mockResolvedValue([]);

    const user = userEvent.setup();
    renderForm();

    const exampleButton = await screen.findByRole("button", { name: /Silvergate Aerospace/ });
    await user.click(exampleButton);

    const entitySelect = screen.getByLabelText("Entity") as HTMLSelectElement;
    await waitFor(() => expect(entitySelect.value).toBe("Silvergate Aerospace"));

    const submit = screen.getByRole("button", { name: /submit erasure request/i });
    expect(submit).toBeEnabled();
  });

  it("disables submit while a job is already active", async () => {
    vi.spyOn(client, "getEntities").mockResolvedValue(ENTITIES);
    vi.spyOn(client, "getAttributes").mockResolvedValue({ company: ["ceo"], person: [] });
    vi.spyOn(client, "getExampleRequests").mockResolvedValue([]);
    const runningJob: JobStatus = {
      job_id: "job-x",
      job_type: "train_and_verify",
      status: "running",
      auto_verify: true,
      created_at: "2026-08-30T12:00:00Z",
      log_tail: [],
    };
    vi.spyOn(client, "listJobs").mockResolvedValue([runningJob]);

    const user = userEvent.setup();
    renderForm();

    const entitySelect = await screen.findByLabelText("Entity");
    await user.selectOptions(entitySelect, "Silvergate Aerospace");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /submit erasure request/i })).toBeDisabled();
    });
    expect(screen.getByText(/A job is already running/)).toBeInTheDocument();
  });
});
