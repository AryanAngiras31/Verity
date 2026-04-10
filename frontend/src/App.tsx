import { useState } from "react";
import { useVerityStore } from "./store/useVerityStore";
import {
  FileText,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  ExternalLink,
} from "lucide-react";

export default function App() {
  const [inputValue, setInputValue] = useState("");
  const {
    currentClaim,
    verdict,
    confidence,
    evidence,
    isLoading,
    verifyClaim,
  } = useVerityStore();

  const handleVerify = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    verifyClaim(inputValue);
  };

  const setExample = (text: string) => {
    setInputValue(text);
  };

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden font-sans">
      {/* LEFT PANE: Input and Verdict */}
      <div className="w-[450px] min-w-[450px] h-full flex flex-col border-r border-border bg-card shadow-sm z-10">
        {/* Header */}
        <header className="p-6 border-b border-border flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
            V
          </div>
          <div>
            <h1 className="text-xl font-semibold leading-tight text-foreground">
              Verity
            </h1>
            <p className="text-sm text-muted-foreground">
              Scientific Fact-Checking Engine
            </p>
          </div>
        </header>

        {/* Verdict Panel */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="bg-background border border-border rounded-xl p-8 flex flex-col items-center justify-center text-center min-h-[250px]">
            {isLoading ? (
              <div className="animate-pulse flex flex-col items-center">
                <div className="w-12 h-12 bg-muted rounded-lg mb-4"></div>
                <div className="h-5 w-32 bg-muted rounded mb-2"></div>
                <div className="h-4 w-48 bg-muted rounded"></div>
              </div>
            ) : verdict ? (
              <>
                <ShieldCheck
                  className={`w-12 h-12 mb-4 ${
                    verdict === "TRUE"
                      ? "text-[hsl(var(--stance-support))]"
                      : verdict === "FALSE"
                        ? "text-[hsl(var(--stance-refute))]"
                        : "text-[hsl(var(--stance-neutral))]"
                  }`}
                />
                <h2 className="text-2xl font-bold mb-2">Verdict: {verdict}</h2>
                <p className="text-muted-foreground">
                  Aggregate Confidence:{" "}
                  <span className="font-semibold text-foreground">
                    {(confidence! * 100).toFixed(1)}%
                  </span>
                </p>
                <p className="text-sm text-muted-foreground mt-4 px-4">
                  Based on {evidence.length} peer-reviewed sources.
                </p>
              </>
            ) : (
              <>
                <div className="w-12 h-12 bg-muted rounded-lg flex items-center justify-center mb-4 text-muted-foreground">
                  <FileText className="w-6 h-6" />
                </div>
                <h2 className="text-lg font-semibold mb-2">Final Verdict</h2>
                <p className="text-sm text-muted-foreground px-4">
                  Enter a scientific or medical claim below to analyze it
                  against peer-reviewed literature.
                </p>
              </>
            )}
          </div>
        </div>

        {/* Claim Input Area */}
        <div className="p-6 border-t border-border bg-card">
          <form onSubmit={handleVerify} className="flex flex-col gap-4">
            <textarea
              className="w-full h-24 p-3 bg-background border border-border rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent text-sm"
              placeholder="Enter a scientific or medical claim to verify..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="w-full bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-medium rounded-xl py-3 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {isLoading ? "Analyzing Literature..." : "Verify Claim"}
              {!isLoading && <ArrowRight className="w-4 h-4" />}
            </button>
          </form>

          <div className="mt-4">
            <p className="text-xs text-muted-foreground mb-2">
              Try an example:
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() =>
                  setExample(
                    "Vitamin D supplementation prevents respiratory infections.",
                  )
                }
                className="text-xs bg-muted hover:bg-border text-muted-foreground px-3 py-1.5 rounded-full transition-colors text-left line-clamp-1"
              >
                Vitamin D supplementation prevents respiratory infections.
              </button>
              <button
                onClick={() =>
                  setExample(
                    "Medications to treat obesity have unwanted side effects.",
                  )
                }
                className="text-xs bg-muted hover:bg-border text-muted-foreground px-3 py-1.5 rounded-full transition-colors text-left line-clamp-1"
              >
                Medications to treat obesity have unwanted side effects.
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT PANE: Evidence Ledger */}
      <div className="flex-1 h-full bg-background flex flex-col overflow-hidden">
        <header className="p-6 border-b border-border bg-background">
          <h2 className="text-lg font-semibold">Evidence Ledger</h2>
          <p className="text-sm text-muted-foreground">
            {currentClaim
              ? `Analyzing: "${currentClaim}"`
              : "Submit a claim to view evidence"}
          </p>
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          {isLoading ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
              <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
              <p>Querying Qdrant Vector Database...</p>
            </div>
          ) : evidence.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-6">
              {evidence.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-card border border-border rounded-xl p-5 shadow-sm hover:shadow transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3 gap-4">
                    <h3 className="font-semibold text-foreground leading-tight">
                      <a
                        href={`https://scholar.google.com/scholar?q=${encodeURIComponent(item.title)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:underline hover:text-primary transition-colors flex items-start gap-1.5 group"
                        title="Search this paper on Google Scholar"
                      >
                        <span>{item.title}</span>
                        <ExternalLink className="w-3.5 h-3.5 mt-1 shrink-0 text-muted-foreground group-hover:text-primary transition-colors" />
                      </a>
                    </h3>
                    <div className="flex flex-col items-end shrink-0">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                          item.stance === "SUPPORT"
                            ? "bg-[hsl(var(--stance-support)/0.15)] text-[hsl(var(--stance-support))]"
                            : item.stance === "REFUTE"
                              ? "bg-[hsl(var(--stance-refute)/0.15)] text-[hsl(var(--stance-refute))]"
                              : "bg-[hsl(var(--stance-neutral)/0.15)] text-[hsl(var(--stance-neutral))]"
                        }`}
                      >
                        {item.stance}
                      </span>
                      <span className="text-xs text-muted-foreground mt-1">
                        {(item.confidence * 100).toFixed(1)}% match
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    "{item.snippet}"
                  </p>
                  <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-muted-foreground">
                    <AlertCircle className="w-3 h-3" />
                    Source:{" "}
                    <span className="uppercase font-medium">{item.source}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto">
              <div className="w-16 h-16 bg-muted rounded-2xl flex items-center justify-center mb-6 text-muted-foreground">
                <FileText className="w-8 h-8" />
              </div>
              <h2 className="text-xl font-semibold mb-2">No Evidence Yet</h2>
              <p className="text-muted-foreground">
                Submit a claim to retrieve and analyze evidence from
                peer-reviewed literature.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
