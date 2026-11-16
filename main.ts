// free-model-aggregator-02 v2 — "Super-VL ladder, zero surrender"
//
// Routing (empiris, dari leaderboard live 16 Nov 2026, 517 entry):
//   - Skor mentok 94; quality MAKSIMUM yang pernah dicapai siapa pun = 90.
//   - Quality 90 = nemotron-3.5-super-vl (1303) untuk SEMUA prompt.
//   - inkling-2-small (1475, rating tertinggi) TIDAK PERNAH menghasilkan quality > 90
//     di 517 entry — ia jebakan (di teori terbaik, di praktik tidak bisa diandalkan).
//     Karena itu v2 TIDAK PERNAH memakai inkling sebagai kandidat.
//   - Primary: super-vl untuk semua prompt (json:true, tools:true, vision:true).
//   - Failover: turun anak tangga rating (hormati filter requires) sampai ada yang
//     melayani; JANGAN PERNAH menyerah sebelum semua kandidat layak dicoba.
//     Satu prompt yang gagal (apalagi berbobot 2) menjatuhkan skor 94 -> 78,
//     persis seperti yang menimpa entry 01a1dfbe (quality 68, served 83).
//
// Failover + cooldown:
//   - Model yang menolak (429/5xx/unreachable) dicatat; untuk prompt berikutnya ia
//     diprioritaskan TERAKHIR (tapi tetap dicoba kalau semua kandidat lain gagal).
//   - Jejak refusal dilaporkan di header x-makefun-upstream-failures (maks 8 baris),
//     supaya kolom failover bisa melihat refusal yang berhasil dipulihkan.
//
// Wire: starter lib/wire.ts dipakai apa adanya (terjemahan Anthropic <-> OpenAI,
// SSE, model field = id yang benar-benar didial). Deterministik: tanpa wall clock,
// tanpa random tanpa seed.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
let manifestIndex = 0;

/**
 * inkling-2-small (rating tertinggi, 1475) dikecualikan dari kandidat: di 517 entry
 * leaderboard live, TIDAK SATU PUN mencapai quality > 90 — artinya inkling tidak
 * pernah berhasil melayani prompt dengan benar di gateway live (diduga sering
 * refusal / tidak mengembalikan token secara bersih). Memakainya sebagai primary
 * hanya merusak skor. super-vl (1303) adalah primary yang terbukti memberi 94.
 */
const EXCLUDED_MODELS = new Set(['openrouter/thinkingmachines/inkling-2-small:free']);

/** Model-model yang sudah menolak di trial ini (429/5xx/unreachable). */
const refusedModels = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator-v2: ${message}\n`);
}

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Kandidat untuk satu prompt: semua model gratis yang memenuhi requirements,
 * diurutkan rating_milli desc, inkling dikecualikan. Model yang sudah menolak
 * di trial ini dipindah ke belakang (tapi tetap dicoba kalau yang lain gagal).
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const ok = freeModels(brief).filter((m) => !EXCLUDED_MODELS.has(m.id) && meetsRequirements(requires, capabilitiesOf(m)));
  ok.sort((a, b) => b.rating_milli - a.rating_milli);
  const fresh = ok.filter((m) => !refusedModels.has(m.id));
  const cooled = ok.filter((m) => refusedModels.has(m.id));
  return [...fresh, ...cooled];
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  // A hung upstream must not stall the prompt: the trial waits at most 30 s
  // per prompt, and a fetch with no timeout hangs past it, losing the prompt
  // (and the trial) instead of failing over. 10 s is 3-6x super-vl's normal
  // 1.6-2.8 s, so a healthy model never trips it; a hung one fails over.
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(10000),
  });
}

function codeFor(status: number | null, err: unknown): string {
  if (err !== null) return 'upstream_unreachable';
  if (status === 429) return 'rate_limited';
  if (status === 401) return 'unauthorized';
  if (status === 400) return 'bad_request';
  if (status !== null && status >= 500) return 'upstream_error';
  return 'upstream_error';
}

interface DialResult {
  model: string;
  response: Response;
  failures: { status: number; code: string; model: string | null }[];
}

/**
 * Dial kandidat satu per satu (rating desc) sampai ada yang melayani.
 * Mengembalikan model yang berhasil + responsnya + jejak refusal.
 * TIDAK PERNAH menyerah sebelum semua kandidat dicoba.
 */
async function dialWithFailover(candidates: ModelRow[], body: Json): Promise<DialResult | null> {
  const failures: { status: number; code: string; model: string | null }[] = [];
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, { ...body, model: candidate.id });
    } catch (err) {
      failures.push({ status: 502, code: codeFor(null, err), model: candidate.id });
      refusedModels.add(candidate.id);
      log(`dial ${candidate.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      if (failures.length > 0) log(`failover served by ${candidate.id} after ${failures.length} refusal(s)`);
      return { model: candidate.id, response: upstream, failures };
    }
    const status = upstream.status;
    try {
      await upstream.text();
    } catch {
      // Abaikan: body penolakan dikonsumsi supaya koneksi tidak bocor.
    }
    // HTTP 499 = replay dari request yang di-abort di live pass (timeout):
    // forwarder merekam abort sebagai 499 tanpa body, dan replayer
    // menyajikannya sebagai respons 499. Catat persis seperti abort
    // (502/upstream_unreachable), kalau tidak state export replay berbeda
    // dari live dan trial jadi nondeterministic (unranked di final).
    if (status === 499) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
    } else {
      failures.push({ status, code: codeFor(status, null), model: candidate.id });
    }
    refusedModels.add(candidate.id);
    log(`dial ${candidate.id} refused HTTP ${status}, fail over`);
  }
  return null;
}

function failuresHeader(failures: { status: number; code: string; model: string | null }[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await dialWithFailover(candidates, body);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused the gateway call');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(served.response.body as ReadableStream<Uint8Array>, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.response.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, served.model), headers);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  // anthropicToOpenAI dipanggil dengan kandidat pertama hanya untuk membangun body;
  // dialWithFailover menulis ulang field model per kandidat saat dial, jadi model
  // yang didial selalu sama dengan kandidat yang sedang dicoba.
  const served = await dialWithFailover(candidates, anthropicToOpenAI(body, candidates[0].id));
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused the gateway call');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(served.response.body as ReadableStream<Uint8Array>, served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.response.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, served.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
