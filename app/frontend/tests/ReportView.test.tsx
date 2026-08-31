import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { WhatWasTargeted } from "../src/components/report/WhatWasTargeted";
import { WhatWasDone } from "../src/components/report/WhatWasDone";
import { VerificationResults } from "../src/components/report/VerificationResults";
import { ImpactAssessment } from "../src/components/report/ImpactAssessment";
import { KeyTakeaways } from "../src/components/report/KeyTakeaways";
import { RawReportJson } from "../src/components/report/RawReportJson";
import type { ErasureReport } from "../src/api/types";
import fixture from "./fixtures/report.revision-1.json";

// Cast through unknown: the fixture is hand-built today (Module 3 hasn't produced
// a real revision-1 yet -- see plan.md's Module 6 Step 8) but its shape is meant to
// be swapped for a REAL captured verification/reports/revision-1_verification_report.json
// the moment one exists, without this test file changing.
const report = fixture as unknown as ErasureReport;

describe("Report section components render the full ErasureReport shape", () => {
  it("WhatWasTargeted shows the request and resolved forget scope", () => {
    render(<WhatWasTargeted report={report} />);
    expect(screen.getByText("Silvergate Aerospace")).toBeInTheDocument();
    expect(screen.getByText("entity")).toBeInTheDocument();
    expect(screen.getByText(/5 fact\(s\) across 1 entity group\(s\)/)).toBeInTheDocument();
  });

  it("WhatWasDone shows method and parent revision", () => {
    render(<WhatWasDone report={report} />);
    expect(screen.getByText("NPO + neighbor-weighted retain")).toBeInTheDocument();
    expect(screen.getByText("revision-0")).toBeInTheDocument();
  });

  it("VerificationResults renders every sub-signal, including the two edge cases", () => {
    render(<VerificationResults vr={report.verification_results} />);
    // direct QA before/after table
    expect(screen.getByText("Forget set")).toBeInTheDocument();
    // reference_model_comparison.available === false must show the reason, not a blank section
    expect(screen.getByText(/Not available for this request/)).toBeInTheDocument();
    // membership_inference.summary.small_forget_set_caveat === true must be surfaced
    expect(screen.getByText("small forget-set caveat applies")).toBeInTheDocument();
    // decoy check renders
    expect(screen.getByText("correctly retained")).toBeInTheDocument();
  });

  it("ImpactAssessment shows pass/fail badges next to the numbers they came from", () => {
    render(<ImpactAssessment impact={report.impact_assessment} />);
    expect(screen.getByText("collapsed (genuinely forgotten)")).toBeInTheDocument();
    expect(screen.getAllByText("within tolerance").length).toBe(2);
  });

  it("KeyTakeaways renders every bullet report.py wrote", () => {
    render(<KeyTakeaways takeaways={report.key_takeaways} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(report.key_takeaways.length);
  });

  it("RawReportJson is present as a fallback and contains the full report", () => {
    render(<RawReportJson report={report} />);
    expect(screen.getByText("Raw report JSON (full fidelity)")).toBeInTheDocument();
  });
});
