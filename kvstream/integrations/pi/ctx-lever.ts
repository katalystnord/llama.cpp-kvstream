// Context lever (idea: David Sandquist, 2026-10-28): "when you load a model you also get to set the context window, with a clear
// indication of what fits on the card and what crosses over into main RAM". Written to grow into something general
// (his words: "eventually should be in Pi itself, once the stream exists for more things"): the models are profiles in a
// config file, and everything the lever shows about a model is read from its own GGUF header, not hard-coded.
//
// Two llama.cpp servers, never both on the GPU:
//   - the everyday router on 8080 (llama-server.service via llama-proxy.socket): qwen38 48K, saluki, cadquery-coder;
//   - the long-context kv-stream fork on 8090 (llama-long.service, katalystnord/llama.cpp-kvstream): one model with the
//     KV cache streamed between a GPU pool and host RAM. Model, window and V cache type live in ~/.config/llama-long.env,
//     which this lever writes and the unit reads (serve-long.sh takes MODEL, MMPROJ, TEMPLATE and SPEC from it).
// Profiles: ~/.config/llama-long/models.json (example: kvstream/integrations/pi/long-models.example.json). Each one is
// registered as "beast-long/<id>-long". /ctx opens the popup (also automatically when a long model is picked and its
// server is down or serving another model); Enter restarts the long server; picking a router model stops it again
// (its unit then restores port 8080); picking an Ollama model (beast-ollama) stops both llama.cpp servers first.
//
// Memory model, measured on an RX 9070 16 GB, 2026-10-28 (kvstream/README.md): KV bytes per token = attention layers x KV heads x
// (K head size x 34/32 + V head size x 34/32 for q8_0 or 18/32 for q4_0), e.g. Qwen3.8 34,816 B (q8 V) and 26,624 B
// (q4 V); the stream keeps a pool on the GPU (--kv-resident-mib-decode: 2420 MiB, or the profile's poolMib for models
// with smaller weights), the rest goes to pinned host RAM
// on top of about 1.1 GB of server overhead. The fork's paged kernels take head size 256 with 6 query heads per KV
// head (pd_shapes_ok in ggml/src/ggml-cuda/fattn-paged-decode.cu); the lever checks every model's header against that.
// Provider names used here (beast-long, beast-llamacpp, beast-ollama) are those of the author's Pi setup, named after
// the test machine; rename them to match your own ~/.pi/agent/models.json.
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import { closeSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";

const HOME = homedir();
const ENV_FILE = HOME + "/.config/llama-long.env";
const PROFILES_FILE = HOME + "/.config/llama-long/models.json";
// Window steps; each model also gets its trained maximum and the largest window whose cache fits in its GPU pool.
const STEPS = [32768, 49152, 65536, 98304, 131072, 163840, 204800, 262144];
const POOL_MIB = 2420; // default --kv-resident-mib-decode in serve-long.sh; a profile may set its own (poolMib)
const WORK_GB = 0.3; // compute buffers beside weights and pool (Qwen3.8: 14.4 GB used with 12.1 GB weights)
const BASE_RAM_GB = 1.1;
const KERNEL = { headSize: 256, gqa: 6 }; // the fork's paged kernel shape

type Speed = [number, number, number]; // [context tokens, write t/s, read t/s] measured near a full window, q8 V, 220 W
interface Profile {
	id: string; name: string; gguf: string; mmproj?: string; template?: string; spec?: string;
	input?: string[]; reasoning?: boolean; speed?: Speed[]; note?: string; poolMib?: number; steps?: number[];
}
interface Shape { attnLayers: number; kvHeads: number; heads: number; keyLen: number; valLen: number; ctxMax: number; arch: string }
interface Model extends Profile { weightsGb: number; shape?: Shape; problem?: string }

// Built-in profile, used when the profiles file is missing: the long server as it was first set up.
const DEFAULT_PROFILES: Profile[] = [{
	id: "qwen38", name: "Qwen3.8 27B",
	gguf: HOME + "/models/Qwen3.8-27B-IQ3_S.gguf",
	speed: [[49152, 60, 640], [65536, 52.5, 577], [131072, 34, 464], [204800, 16.4, 379]],
}];

// GGUF metadata reader: the key/value header only (tensor data is never read). Arrays of strings (the tokenizer) are
// skipped in place; small numeric arrays are kept, since some models store per-layer head counts.
function ggufMeta(path: string): Record<string, any> {
	const fd = openSync(path, "r");
	let pos = 0, buf = Buffer.alloc(0), bufStart = 0;
	const at = (n: number): number => { // byte offset in buf of the next n bytes; advances pos
		if (pos < bufStart || pos + n > bufStart + buf.length) {
			const b = Buffer.alloc(Math.max(n, 1 << 20));
			const got = readSync(fd, b, 0, b.length, pos);
			if (got < n) throw new Error("GGUF header cut short");
			buf = b.subarray(0, got); bufStart = pos;
		}
		const off = pos - bufStart; pos += n; return off;
	};
	const rd = <T>(n: number, f: (b: Buffer, o: number) => T): T => { const o = at(n); return f(buf, o); }; // at() first: it may swap buf
	const u32 = () => rd(4, (b, o) => b.readUInt32LE(o));
	const u64 = () => rd(8, (b, o) => Number(b.readBigUInt64LE(o)));
	const str = () => { const n = u64(); return rd(n, (b, o) => b.toString("utf8", o, o + n)); };
	const SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
	const scalar = (t: number): any => {
		switch (t) {
			case 0: return rd(1, (b, o) => b.readUInt8(o)); case 1: return rd(1, (b, o) => b.readInt8(o));
			case 2: return rd(2, (b, o) => b.readUInt16LE(o)); case 3: return rd(2, (b, o) => b.readInt16LE(o));
			case 4: return u32(); case 5: return rd(4, (b, o) => b.readInt32LE(o));
			case 6: return rd(4, (b, o) => b.readFloatLE(o)); case 7: return rd(1, (b, o) => b.readUInt8(o) !== 0);
			case 8: return str();
			case 10: return u64(); case 11: return rd(8, (b, o) => Number(b.readBigInt64LE(o)));
			case 12: return rd(8, (b, o) => b.readDoubleLE(o));
		}
		throw new Error("GGUF value type " + t);
	};
	try {
		if (rd(4, (b, o) => b.toString("latin1", o, o + 4)) !== "GGUF") throw new Error("not a GGUF file");
		u32(); u64(); // version, tensor count
		const n = u64(), out: Record<string, any> = {};
		for (let k = 0; k < n; k++) {
			const key = str(), t = u32();
			if (t !== 9) { out[key] = scalar(t); continue; }
			const et = u32(), len = u64();
			if (et === 8 || et === 9) { for (let j = 0; j < len; j++) { if (et === 8) str(); else throw new Error("nested GGUF array"); } }
			else if (len <= 1024) out[key] = Array.from({ length: len }, () => scalar(et));
			else pos += len * SIZE[et];
		}
		return out;
	} finally { closeSync(fd); }
}

function shapeOf(gguf: string): Shape {
	const m = ggufMeta(gguf), a = m["general.architecture"];
	const num = (k: string, d = 0) => { const v = m[`${a}.${k}`]; return Array.isArray(v) ? Math.max(...v) : (v ?? d); };
	const heads = num("attention.head_count"), kvHeads = num("attention.head_count_kv", heads);
	const keyLen = num("attention.key_length", num("embedding_length") / heads);
	const blocks = num("block_count") - num("nextn_predict_layers"), interval = num("full_attention_interval", 1);
	return { arch: a, heads, kvHeads, keyLen, valLen: num("attention.value_length", keyLen),
		attnLayers: Math.floor(blocks / interval), ctxMax: num("context_length") };
}

function bytesPerToken(s: Shape, ctv: string): number {
	return s.attnLayers * s.kvHeads * (s.keyLen * 34 / 32 + s.valLen * (ctv === "q4_0" ? 18 : 34) / 32);
}

function poolBytes(m: Model): number { return (m.poolMib ?? POOL_MIB) * 1048576; }

// Largest window (a multiple of 4096) whose whole cache fits in the model's GPU pool: nothing streams from RAM.
function onCardTokens(m: Model, ctv: string): number {
	return m.shape ? Math.floor(poolBytes(m) / bytesPerToken(m.shape, ctv) / 4096) * 4096 : 0;
}

function stepsFor(m: Model, ctv: string): number[] {
	const max = m.shape?.ctxMax || STEPS[STEPS.length - 1];
	const all = new Set((m.steps ?? STEPS).filter((s) => s <= max));
	const onCard = onCardTokens(m, ctv);
	if (onCard >= 8192 && onCard < max) all.add(onCard);
	all.add(max);
	return [...all].sort((a, b) => a - b);
}

function sizeGb(path?: string): number { try { return path ? statSync(path).size / 1e9 : 0; } catch { return 0; } }

function loadModels(): Model[] {
	let profiles = DEFAULT_PROFILES;
	try { profiles = JSON.parse(readFileSync(PROFILES_FILE, "utf8")).models; } catch {}
	return profiles.map((p) => {
		const m: Model = { ...p, weightsGb: sizeGb(p.gguf) + sizeGb(p.mmproj) };
		try {
			const s = (m.shape = shapeOf(p.gguf));
			if (s.keyLen !== KERNEL.headSize || s.valLen !== KERNEL.headSize) m.problem = `head size ${s.keyLen}/${s.valLen}; the stream kernels need ${KERNEL.headSize}`;
			else if (s.heads !== KERNEL.gqa * s.kvHeads) m.problem = `${s.heads} query heads over ${s.kvHeads} KV heads; the stream kernels need ${KERNEL.gqa} per KV head`;
		} catch (e: any) { m.problem = "cannot read the GGUF: " + (e?.message ?? e); }
		return m;
	});
}

function readEnv(): { ctx: number; ctv: string; id: string } {
	let ctx = 98304, ctv = "q8_0", id = "qwen38";
	try {
		for (const line of readFileSync(ENV_FILE, "utf8").split("\n")) {
			const m = line.match(/^\s*(CTX|CTV|LONG_MODEL)\s*=\s*(\S+)/);
			if (m && m[1] === "CTX") ctx = Number(m[2]) || ctx;
			if (m && m[1] === "CTV") ctv = m[2];
			if (m && m[1] === "LONG_MODEL") id = m[2];
		}
	} catch {}
	return { ctx, ctv, id };
}

function writeEnv(m: Model, ctx: number, ctv: string) {
	// systemd EnvironmentFile: double quotes keep SPEC's spaces; empty values fall back to serve-long.sh's defaults.
	writeFileSync(ENV_FILE, [
		"# Written by the Pi context lever (/ctx); read by llama-long.service and serve-long.sh.",
		`LONG_MODEL=${m.id}`, `CTX=${ctx}`, `CTV=${ctv}`, `MODEL=${m.gguf}`, `MMPROJ=${m.mmproj ?? ""}`,
		`TEMPLATE=${m.template ?? ""}`, `SPEC="${m.spec ?? ""}"`, `POOL_MIB=${m.poolMib ?? POOL_MIB}`, "",
	].join("\n"));
}

function speed(m: Model, ctx: number, ctv: string): { write: number; read: number; guess: boolean } | undefined {
	const t = m.speed;
	if (!t || t.length === 0) return undefined;
	let w = t[0][1], r = t[0][2];
	if (ctx >= t[t.length - 1][0]) {
		const [c1, w1, r1] = t[t.length - 1];
		w = w1 * (c1 / ctx); r = r1 * (c1 / ctx); // beyond the last measurement: assume it falls with the window
	} else {
		for (let i = 0; i + 1 < t.length; i++) {
			const [c0, w0, r0] = t[i], [c1, w1, r1] = t[i + 1];
			if (ctx >= c0 && ctx <= c1) {
				const f = (ctx - c0) / (c1 - c0);
				w = w0 + f * (w1 - w0); r = r0 + f * (r1 - r0);
				break;
			}
		}
	}
	if (ctv === "q4_0") { w *= 1.25; r *= 1.2; } // q4 V wrote 8-35% faster at depth (Qwen3.8)
	// Past the last measurement the fall is steeper than this (Saluki 250K with a small pool wrote at 3.9 t/s): a guess.
	return { write: w, read: r, guess: ctx > t[t.length - 1][0] * 1.05 };
}

function gpuTotalGb(): number {
	for (const c of ["card1", "card0"]) { try { return Number(readFileSync(`/sys/class/drm/${c}/device/mem_info_vram_total`, "utf8")) / 1073741824; } catch {} }
	return 16;
}
function ramTotalGb(): number {
	try { return Number(readFileSync("/proc/meminfo", "utf8").match(/MemTotal:\s+(\d+)/)![1]) / 1048576; } catch { return 60; }
}

export default function ctxLever(pi: ExtensionAPI) {
	let models = loadModels();
	const GPU_GB = Math.round(gpuTotalGb()), RAM_GB = Math.round(ramTotalGb());
	const usable = () => models.filter((m) => !m.problem);

	function registerLong(ctx: number) {
		pi.registerProvider("beast-long", {
			name: "Long context (kv-stream)",
			baseUrl: "http://127.0.0.1:8090/v1",
			api: "openai-completions",
			apiKey: "none",
			models: usable().map((m) => ({
				id: `${m.id}-long`,
				name: `${m.name} long (${Math.round(ctx / 1024)}K, cache streams to RAM)`,
				reasoning: m.reasoning ?? true,
				input: m.input ?? ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: ctx,
				maxTokens: 8192,
				compat: { thinkingFormat: "qwen-chat-template" },
			})) as any,
		} as any);
	}

	async function longActive(): Promise<boolean> {
		const r = await pi.exec("systemctl", ["--user", "is-active", "llama-long.service"]);
		return r.stdout.trim() === "active";
	}

	type Choice = { model: Model; ctx: number; ctv: string };
	async function lever(ctx: ExtensionContext, preselect?: string): Promise<Choice | undefined> {
		models = loadModels(); // pick up profile edits without a restart
		const cur = readEnv();
		let mi = Math.max(0, models.findIndex((m) => m.id === (preselect ?? cur.id)));
		let ctv = cur.ctv === "q4_0" ? "q4_0" : "q8_0";
		let tokens = cur.ctx;
		// Keep the window on a step of the current model; stay on the "all on the card" step when the model or V type
		// changes it.
		let pinned = models[mi] && tokens === onCardTokens(models[mi], ctv);
		const snap = () => {
			const steps = stepsFor(models[mi], ctv);
			if (pinned && onCardTokens(models[mi], ctv) >= 8192) tokens = onCardTokens(models[mi], ctv);
			else if (!steps.includes(tokens)) tokens = [...steps].reverse().find((s) => s <= tokens) ?? steps[0];
			return steps;
		};
		const move = (dir: number) => {
			const steps = snap();
			tokens = dir < 0 ? ([...steps].reverse().find((s) => s < tokens) ?? steps[0]) : (steps.find((s) => s > tokens) ?? steps[steps.length - 1]);
			pinned = tokens === onCardTokens(models[mi], ctv);
		};
		return ctx.ui.custom<Choice | undefined>((tui, theme, _kb, done) => {
			const bar = (parts: [number, string, string][], total: number, width: number) => {
				let out = "", used = 0;
				for (const [gb, colour, ch] of parts) {
					const n = Math.max(gb > 0 ? 1 : 0, Math.round((gb / total) * width));
					out += theme.fg(colour as any, ch.repeat(Math.min(n, width - used)));
					used += Math.min(n, width - used);
				}
				return out + theme.fg("dim", "░".repeat(Math.max(0, width - used)));
			};
			return {
				render(width: number): string[] {
					// A framed box with a solid background, so the transcript behind it does not show through.
					const inner = Math.max(40, Math.min(width - 4, 74));
					snap();
					const m = models[mi], barW = inner - 12;
					const pick = models.length > 1 ? theme.fg("dim", "▲▼ ") : "";
					const modelLine = "model    " + pick + theme.bold(m.name) +
						theme.fg("dim", `  ${m.weightsGb.toFixed(1)} GB${m.mmproj ? ", with vision" : ""}${m.note ? " · " + m.note : ""}`);
					let body: string[];
					if (m.problem || !m.shape) {
						body = ["", modelLine, "", theme.fg("error", "cannot stream: " + (m.problem ?? "unknown shape")), "",
							theme.fg("dim", "↑/↓ model · Esc cancel"), ""];
					} else {
						const s = m.shape, bpt = bytesPerToken(s, ctv);
						const kvGb = (tokens * bpt) / 1e9;
						const poolGb = poolBytes(m) / 1e9;
						const onGpu = Math.min(kvGb, poolGb), inRam = Math.max(0, kvGb - poolGb);
						const sp = speed(m, tokens, ctv);
						const over = tokens > s.ctxMax ? theme.fg("warning", `  over the model's trained ${Math.round(s.ctxMax / 1024)}K`) : "";
						body = [
							"",
							modelLine,
							"window   " + theme.fg("dim", "◀ ") + theme.bold(`${Math.round(tokens / 1024)}K tokens`) + theme.fg("dim", " ▶") +
								theme.fg("dim", `    V cache ${ctv === "q8_0" ? "q8 (coding quality)" : "q4 (faster, not for coding)"}`) + over +
								(tokens === onCardTokens(m, ctv) ? theme.fg("success", "  largest all on the card") : ""),
							theme.fg("dim", `         ${s.attnLayers} attention layers x ${s.kvHeads} KV heads x ${s.keyLen}: ${Math.round(bpt).toLocaleString("en")} B per token`),
							"",
							`GPU ${GPU_GB} GB `.padEnd(10) + bar([[m.weightsGb, "muted", "█"], [WORK_GB, "dim", "▓"], [onGpu, "success", "█"]], GPU_GB, barW),
							theme.fg("dim", `          weights ${m.weightsGb.toFixed(1)} GB · work ${WORK_GB.toFixed(1)} GB · `) +
								theme.fg("success", `cache on card ${onGpu.toFixed(1)} GB`),
							`RAM ${RAM_GB} GB `.padEnd(10) + bar([[BASE_RAM_GB, "dim", "▓"], [inRam, "warning", "█"]], RAM_GB, barW),
							theme.fg("dim", `          server ${BASE_RAM_GB.toFixed(1)} GB · `) +
								(inRam > 0 ? theme.fg("warning", `streamed from RAM ${inRam.toFixed(1)} GB`) : theme.fg("success", "nothing streamed: all on the card")),
							"",
							sp ? `full window: write ~${Math.round(sp.write)} t/s, read ~${Math.round(sp.read)} t/s` : theme.fg("dim", "speed: not measured for this model yet"),
							sp ? `reading a full window takes ~${Math.max(1, Math.round(tokens / sp.read / 60))} min` + theme.fg(sp.guess ? "warning" : "dim", sp.guess ? " (beyond the measurements: a guess, likely slower)" : " (measured on an RX 9070, 220 W)") : "",
							"",
							theme.fg("dim", "↑/↓ model · ←/→ window · q q8/q4 · Enter apply (restarts the server) · Esc cancel"),
							"",
						];
					}
					const fill = (l: string) => {
						const t = truncateToWidth(l, inner);
						return t + " ".repeat(Math.max(0, inner - visibleWidth(t)));
					};
					const title = truncateToWidth(` Context lever: ${m.name} long (kv-stream) `, inner);
					const top = theme.fg("accent", "╭─") + theme.fg("accent", theme.bold(title)) +
						theme.fg("accent", "─".repeat(Math.max(0, inner + 1 - visibleWidth(title))) + "╮"); // top = inner + 4 wide, like the rows
					const rows = body.map((l) => theme.fg("accent", "│") + " " + fill(l) + " " + theme.fg("accent", "│"));
					const bottom = theme.fg("accent", "╰" + "─".repeat(inner + 2) + "╯");
					return [top, ...rows, bottom].map((l) => theme.bg("customMessageBg" as any, l));
				},
				handleInput(data: string): void {
					if (data === "\x1b[D" || data === "h") move(-1);
					else if (data === "\x1b[C" || data === "l") move(1);
					else if (data === "\x1b[A" || data === "k") mi = (mi + models.length - 1) % models.length;
					else if (data === "\x1b[B" || data === "j") mi = (mi + 1) % models.length;
					else if (data === "q") ctv = ctv === "q8_0" ? "q4_0" : "q8_0";
					else if (data === "\r" || data === "\n") { if (!models[mi].problem) done({ model: models[mi], ctx: tokens, ctv }); return; }
					else if (data === "\x1b" || data === "\x03") { done(undefined); return; }
					else return;
					tui.requestRender();
				},
				invalidate() {},
			};
		}, { overlay: true, overlayOptions: { width: 80, anchor: "center" } as any });
	}

	async function applyChoice(choice: Choice, ctx: ExtensionContext) {
		const { model: m } = choice;
		writeEnv(m, choice.ctx, choice.ctv);
		registerLong(choice.ctx);
		ctx.ui.notify(`Starting ${m.name} long with ${Math.round(choice.ctx / 1024)}K (${choice.ctv}); about 10 s...`, "info");
		const r = await pi.exec("systemctl", ["--user", "restart", "llama-long.service"], { timeout: 700000 } as any);
		if (r.code !== 0) {
			ctx.ui.notify("Long server failed to start: " + (r.stderr || r.stdout).slice(0, 200), "error");
			return;
		}
		const model = (ctx as any).modelRegistry?.find?.("beast-long", `${m.id}-long`);
		if (model) await pi.setModel(model);
		ctx.ui.notify(`Long context ready: ${m.name}, ${Math.round(choice.ctx / 1024)}K on port 8090; the 8080 router is paused.`, "info");
	}

	registerLong(readEnv().ctx); // at load, so the long models are listed everywhere (also --list-models)
	pi.on("session_start", async () => {
		models = loadModels();
		registerLong(readEnv().ctx); // re-read the env file: the window may have changed since load
	});

	pi.registerCommand("ctx", {
		description: "Context lever for the long-context models: model, window size, what fits on the GPU, what streams to RAM",
		handler: async (_args, ctx) => {
			if (!ctx.hasUI) return;
			const choice = await lever(ctx);
			if (choice) await applyChoice(choice, ctx);
		},
	});

	pi.on("model_select", async (event, ctx) => {
		const provider = (event.model as any).provider;
		if (provider === "beast-long") {
			// Open the lever when the long server is down or serving a different model than the one picked.
			const id = String((event.model as any).id).replace(/-long$/, "");
			if (ctx.hasUI && (!(await longActive()) || readEnv().id !== id)) {
				const choice = await lever(ctx, id);
				if (choice) await applyChoice(choice, ctx);
			}
		} else if (provider === "beast-ollama") {
			// Ollama loads its own copy on the GPU: stop both llama.cpp servers first (the 8080 socket stays, and the
			// router unloads Ollama's models again when it next starts).
			await pi.exec("systemctl", ["--user", "stop", "llama-long.service", "llama-server.service", "llama-proxy.service"]);
			await pi.exec("systemctl", ["--user", "start", "llama-proxy.socket"]);
			if (ctx.hasUI) ctx.ui.notify("llama.cpp servers stopped so Ollama has the GPU.", "info");
		} else if (provider === "beast-llamacpp" && (await longActive())) {
			// Back to the router: stop the long server; its unit starts the 8080 socket again.
			await pi.exec("systemctl", ["--user", "stop", "llama-long.service"]);
			if (ctx.hasUI) ctx.ui.notify("Long server stopped; the 8080 router is back.", "info");
		}
	});
}
