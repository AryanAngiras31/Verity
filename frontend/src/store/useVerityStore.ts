import { create } from "zustand";

// 1. Map to your Rust backend types
export interface Evidence {
  title: string;
  source: string;
  snippet: string;
  stance: string;
  confidence: number;
}

export interface VerifyResponse {
  final_verdict: string;
  aggregate_confidence: number;
  evidence: Evidence[];
}

// 2. Define the Zustand Store Interface
interface VerityState {
  // --- Data State ---
  currentClaim: string | null;
  verdict: string | null;
  confidence: number | null;
  evidence: Evidence[];

  // --- UI State ---
  isLoading: boolean;
  error: string | null;

  // --- Actions ---
  verifyClaim: (claim: string) => Promise<void>;
  reset: () => void;
}

// 3. Create the Store
export const useVerityStore = create<VerityState>((set) => ({
  // Initial State
  currentClaim: null,
  verdict: null,
  confidence: null,
  evidence: [],
  isLoading: false,
  error: null,

  reset: () =>
    set({
      currentClaim: null,
      verdict: null,
      confidence: null,
      evidence: [],
      error: null,
      isLoading: false,
    }),

  // Async Action to call the Rust API
  verifyClaim: async (claim: string) => {
    set({
      isLoading: true,
      error: null,
      currentClaim: claim,
      // Clear previous results while loading
      verdict: null,
      confidence: null,
      evidence: [],
    });

    try {
      // The Vite proxy will route this to http://backend:8080/api/verify
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          claim,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const data: VerifyResponse = await response.json();

      set({
        isLoading: false,
        verdict: data.final_verdict,
        confidence: data.aggregate_confidence,
        evidence: data.evidence,
      });
    } catch (error: any) {
      set({
        isLoading: false,
        error:
          error.message ||
          "Failed to verify claim. Did not receive a response from the server.",
      });
    }
  },
}));
