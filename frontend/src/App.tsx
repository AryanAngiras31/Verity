import { useState, useEffect } from "react";
import { useVerityStore } from "./store/useVerityStore";
import {
  FileText,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  ExternalLink,
  XCircle,
  X
} from "lucide-react";
import logo from "./assets/logo.png";

// The actual steps your Rust backend performs
const LOADING_STEPS = [
  "Embedding claim...",
  "Querying the database...",
  "Retrieving semantic matches...",
  "Running the cross-encoder...",
  "Analyzing logical entailment...",
  "Aggregating evidence stance...",
];

export default function App() {
  const [inputValue, setInputValue] = useState("");
  const [messageIndex, setMessageIndex] = useState(0);

  const {
    currentClaim,
    verdict,
    confidence,
    evidence,
    isLoading,
    verifyClaim,
    error,
    clearError,
  } = useVerityStore();

  // Cycle through loading messages every 2 seconds while fetching
  useEffect(() => {
    let interval: number;
    if (isLoading) {
      interval = window.setInterval(() => {
        setMessageIndex((prev) => (prev + 1) % LOADING_STEPS.length);
      }, 4000);
    } else {
      setMessageIndex(0);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

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
      <div className="w-[450px] min-w-[450px] h-full flex flex-col border-r border-border bg-background shadow-sm z-10">
        <header className="h-24 p-6 border-b border-border flex items-center gap-3 shrink-0">
          <img
            src={logo}
            alt="Verity Logo"
            className="w-11 h-11 rounded-full object-cover shadow-sm border border-border/50"
          />
          <div>
            <h1 className="text-xl font-semibold leading-tight text-foreground">
              Verity
            </h1>
            <p className="text-sm text-muted-foreground">
              Scientific Fact-Checking Engine
            </p>
          </div>
        </header>

        <div className="flex-1 p-6 overflow-y-auto">
          <div className="bg-card border border-border rounded-xl p-8 flex flex-col items-center justify-center text-center min-h-[250px]">
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
                <p className="text-sm text-muted-foreground mt-4 px-4 font-medium italic">
                  "Based on {evidence.length} peer-reviewed sources."
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

        <div className="p-6 border-t border-border bg-background">
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
              className="w-full bg-primary text-foreground hover:opacity-85 transition-opacity font-semibold rounded-xl py-3 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {isLoading ? "Analyzing..." : "Verify Claim"}
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
                  setExample("Mitochondria play a major role in apoptosis.")
                }
                className="text-xs bg-muted hover:bg-border text-muted-foreground px-3 py-1.5 rounded-full transition-colors text-left line-clamp-1"
              >
                Mitochondria play a major role in apoptosis.
              </button>
              <button
                onClick={() =>
                  setExample(
                    "Medications to treat obesity do not have unwanted side effects.",
                  )
                }
                className="text-xs bg-muted hover:bg-border text-muted-foreground px-3 py-1.5 rounded-full transition-colors text-left line-clamp-1"
              >
                Medications to treat obesity do not have unwanted side effects.
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT PANE: Evidence Ledger */}
      <div className="flex-1 h-full bg-background flex flex-col overflow-hidden relative">
        <header className="h-24 p-6 border-b border-border bg-background shrink-0">
          <h2 className="text-lg font-semibold">Evidence Ledger</h2>
          <div className="text-sm text-muted-foreground h-5">
            {currentClaim && (
              <p className="flex items-center gap-1.5">
                <span>Analyzing:</span>
                {/* Subtle Pulse Animation on the claim text */}
                <span
                  className={`italic font-medium text-foreground ${isLoading ? "animate-pulse" : ""}`}
                >
                  "{currentClaim}"
                </span>
              </p>
            )}
            {!currentClaim && <p>Submit a claim to view evidence</p>}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 bg-background">
          {isLoading ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground animate-in fade-in duration-700 bg-background">
              <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-6"></div>
              {/* Cycling Technical Tasks */}
              <p className="text-sm font-medium opacity-80 animate-pulse text-muted-foreground">
                {LOADING_STEPS[messageIndex]}
              </p>
            </div>
          ) : evidence.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-6 animate-in slide-in-from-bottom-4 duration-500">
              {evidence.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-card border border-border rounded-xl p-5 shadow-sm hover:shadow-md transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-start mb-3">
                      <h3 className="font-semibold text-foreground leading-tight">
                        <a
                          href={`https://scholar.google.com/scholar?q=${encodeURIComponent(item.title)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:underline hover:text-blue-400 transition-colors flex items-start gap-1.5 group"
                          title="Search on Google Scholar"
                        >
                          <span>{item.title}</span>
                          <ExternalLink className="w-3.5 h-3.5 mt-1 shrink-0 text-muted-foreground group-hover:text-blue-400" />
                        </a>
                      </h3>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      "{item.snippet}"
                    </p>
                  </div>

                  <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-3 h-3" />
                      <span>
                        Source:{" "}
                        <span className="uppercase font-medium tracking-wider">
                          {item.source}
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded-full font-bold text-[10px] ${
                          item.stance === "SUPPORT"
                            ? "bg-[hsl(var(--stance-support)/0.15)] text-[hsl(var(--stance-support))]"
                            : item.stance === "REFUTE"
                              ? "bg-[hsl(var(--stance-refute)/0.15)] text-[hsl(var(--stance-refute))]"
                              : "bg-[hsl(var(--stance-neutral)/0.15)] text-[hsl(var(--stance-neutral))]"
                        }`}
                      >
                        {item.stance}
                      </span>
                      <span className="font-medium whitespace-nowrap">
                        {(item.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto opacity-60 bg-background">
              <div className="w-16 h-16 bg-muted rounded-2xl flex items-center justify-center mb-6">
                <FileText className="w-8 h-8" />
              </div>
              <h2 className="text-xl font-semibold mb-2">No Evidence Yet</h2>
              <p className="text-muted-foreground text-sm">
                Submit a claim to retrieve and analyze evidence from
                peer-reviewed literature.
              </p>
            </div>
          )}
        </div>
        {error && (
          <div className="absolute top-[55%] left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-lg px-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="bg-card/95 backdrop-blur-md border border-red-500/50 shadow-2xl rounded-2xl p-6 py-8 min-h-[180px] flex items-start gap-4">
              <XCircle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-base font-semibold text-foreground mb-1.5">
                  Connection Error
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  The Verity backend is currently unreachable. The service may be asleep or down. Please try again later.
                </p>
                <p className="text-xs font-mono text-red-400/70 mt-3 line-clamp-2">
                  {error}
                </p>
              </div>
              <button
                onClick={clearError}
                className="text-muted-foreground hover:text-foreground transition-colors p-1.5 rounded-md hover:bg-muted"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
