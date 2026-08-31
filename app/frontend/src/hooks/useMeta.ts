import { useEffect, useState } from "react";
import { getAttributes, getEntities, getExampleRequests } from "../api/client";
import type { AttributesResponse, EntityListItem, ExampleRequest } from "../api/types";

interface UseMetaResult {
  entities: EntityListItem[];
  attributes: AttributesResponse;
  examples: ExampleRequest[];
  loading: boolean;
  error: string | null;
}

const EMPTY_ATTRIBUTES: AttributesResponse = { company: [], person: [] };

/**
 * GET /entities, /attributes, /requests/examples once on mount -- backs
 * ErasureRequestForm's dropdowns and ExampleRequestPicker's buttons. This data is
 * effectively static for the life of a session (it's the fixed dataset schema, not
 * anything a job changes), so unlike useRevisions this never needs a refresh().
 */
export function useMeta(): UseMetaResult {
  const [entities, setEntities] = useState<EntityListItem[]>([]);
  const [attributes, setAttributes] = useState<AttributesResponse>(EMPTY_ATTRIBUTES);
  const [examples, setExamples] = useState<ExampleRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getEntities(), getAttributes(), getExampleRequests()])
      .then(([entitiesRes, attributesRes, examplesRes]) => {
        if (cancelled) return;
        setEntities(entitiesRes);
        setAttributes(attributesRes);
        setExamples(examplesRes);
        setError(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { entities, attributes, examples, loading, error };
}
