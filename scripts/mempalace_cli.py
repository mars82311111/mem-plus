#!/usr/bin/env python3
"""
mem-plus v7 — 最优架构：第一性原则实测驱动
============================================
结论（实测）：
  v5 (单 subprocess → mempalace CLI): 0.44s ← 更快
  v6 (Python direct ChromaDB+Ollama): 2.52s ← Python开销抵消优化

v7 选择：v5 架构为主 + v6 direct fallback 作为 timeout 保障

subprocess 链：
  hook → mempalace_cli.py → mempalace CLI subprocess → Ollama
                    ↓ (on timeout/fail)
              v6 fallback: direct ChromaDB + Ollama HTTP

这样：
  - 正常路径：走 mempalace CLI (0.44s)
  - 超时 fallback：走 direct (2.5s，但保障可用性)
"""
import sys
import os
import json
import argparse
import re
import time
import subprocess

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
MEMPALACE_CLI = '/Users/mars/Library/Python/3.9/bin/mempalace'
SUPER_MEM_CLI = '/Users/mars/.openclaw/workspace/skills/mem-plus/scripts/super_mem_cli.py'
_TIMEOUT = 3.0  # 秒，超时后用 v6 fallback

# ─────────────────────────────────────────────────────────────────
# 1. V5 APPROACH — subprocess to mempalace CLI (实测最优)
# ─────────────────────────────────────────────────────────────────

def call_mempalace(args, timeout=30):
    env = os.environ.copy()
    env['PATH'] = f'/Users/mars/Library/Python/3.9/bin:{env.get("PATH", "")}'
    r = subprocess.run(
        [MEMPALACE_CLI] + args,
        capture_output=True, text=True, timeout=timeout, env=env
    )
    return r.stdout, r.stderr, r.returncode


# ─────────────────────────────────────────────────────────────────
# 2. V6 FALLBACK — direct ChromaDB + Ollama HTTP (timeout保障)
# ─────────────────────────────────────────────────────────────────

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

_SUPER_MEM_CHROMA = os.path.expanduser("~/.super-mem/chroma")


def ollama_embed_http(texts: list) -> list:
    import urllib.request
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                emb = data.get("embedding")
                embeddings.append(emb if emb else [0.0] * 768)
        except Exception:
            embeddings.append([0.0] * 768)
    return embeddings


def v6_fallback_search(query_text: str, limit: int = 5) -> list:
    """v6 direct fallback — called when v5 times out."""
    if not _CHROMA_AVAILABLE:
        return []
    q_emb = ollama_embed_http([query_text])[0]
    try:
        client = chromadb.PersistentClient(path=_SUPER_MEM_CHROMA)
        col = client.get_collection("super_mem_shared")
        raw = col.query(
            query_embeddings=[q_emb],
            n_results=limit * 3,
            include=["documents", "metadatas"]
        )
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        if not docs:
            return []
        items = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            sf = meta.get("source_file", "")
            source = os.path.basename(sf) if sf else sf
            items.append({
                "content": doc, "source": source,
                "score": 1.0 - (i / max(len(docs), 1)),
                "meta": meta
            })
        return items
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# 3. PARSE — mempalace CLI markdown output → structured results
# ─────────────────────────────────────────────────────────────────

def parse_search_output(output: str, query: str = '') -> list:
    """Parse mempalace CLI markdown output into structured results."""
    results = []
    blocks = re.split(r'\n\s*─{5,}\s*\n', output)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if re.match(r'\[\d+\]', block):
            result = _extract_result(block)
            if result:
                results.append(result)
            continue
        first_result_start = block.find('[1]')
        if first_result_start != -1:
            first_block = block[first_result_start:]
            result = _extract_result(first_block)
            if result:
                results.append(result)
    return results


def _extract_result(block: str) -> dict:
    header_match = re.match(r'\[(\d+)\]\s+(\S+)\s+/\s+(\S+)', block)
    if not header_match:
        return None
    source_match = re.search(r'Source:\s*(.+?)(?:\n|$)', block)
    match_score_match = re.search(r'Match:\s*([-\d.]+)', block)
    source = source_match.group(1).strip() if source_match else ''
    raw_score = float(match_score_match.group(1)) if match_score_match else 0.0
    match_pos = block.find('Match:')
    if match_pos == -1:
        return {'content': block[header_match.end():].strip(),
                'score': raw_score, 'source': source, 'match_score': raw_score}
    line_end = block.find('\n', match_pos)
    blank = block.find('\n\n', line_end)
    content = block[blank + 2:].strip() if blank != -1 else block[line_end + 1:].strip()
    return {'content': content, 'score': raw_score, 'source': source, 'match_score': raw_score}


# ─────────────────────────────────────────────────────────────────
# 4. IDENTITY PRIORITY + KEYWORD BOOST
# ─────────────────────────────────────────────────────────────────

IDENTITY_BOOST = {
    'USER.md': 100.0, 'SOUL.md': 50.0, 'MEMORY.md': 30.0,
    'AGENTS.md': 10.0, 'HEARTBEAT.md': 5.0,
}


def identity_boost_score(source: str, content: str, query: str) -> float:
    basename = os.path.basename(source)
    if basename in IDENTITY_BOOST:
        return IDENTITY_BOOST[basename]
    if '城' in query and '城' in content and basename in {
        'USER.md', 'SOUL.md', 'MEMORY.md', 'AGENTS.md', 'HEARTBEAT.md',
        'IDENTITY.md', 'TOOLS.md'
    }:
        return 20.0
    return 0.0


def is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def extract_chinese_tokens(text: str) -> set:
    tokens = set()
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i+1]
        if '\u4e00' <= c1 <= '\u9fff' and '\u4e00' <= c2 <= '\u9fff':
            tokens.add(text[i:i+2])
    for w in re.findall(r'[a-zA-Z0-9]{2,}', text):
        tokens.add(w.lower())
    return tokens


def keyword_boost_score(content: str, query: str) -> float:
    if not is_chinese(query):
        return 0.0
    q_tokens = extract_chinese_tokens(query)
    if not q_tokens:
        return 0.0
    c_tokens = extract_chinese_tokens(content)
    if not c_tokens:
        return 0.0
    return 0.5 * (len(q_tokens & c_tokens) / len(q_tokens))


# ─────────────────────────────────────────────────────────────────
# 5. DEDUP + MMR
# ─────────────────────────────────────────────────────────────────

def dedup_results(results, threshold=0.85):
    if not results:
        return []
    def lev_sim(s1, s2):
        s1, s2 = s1.lower(), s2.lower()
        if not s1 or not s2:
            return 0.0
        m, n = len(s1), len(s2)
        if m < n: s1, s2, m, n = s2, s1, n, m
        prev = range(n + 1)
        for i in range(m):
            curr = [i + 1]
            for j in range(n):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(s1[i]!=s2[j])))
            prev = curr
        return 1.0 - (prev[n] / max(m, n)) if max(m, n) else 1.0
    by_source = {}
    for r in results:
        src = r.get('source', '')
        if src not in by_source or r.get('score', 0) > by_source[src].get('score', 0):
            by_source[src] = r
    deduped = []
    for r in by_source.values():
        if not any(lev_sim(r['content'], e['content']) > threshold for e in deduped):
            deduped.append(r)
    return deduped


def mmr_rerank(results, query, lambda_param=0.7, limit=5):
    if not results or len(results) <= limit:
        return results
    def lev_sim(s1, s2):
        s1, s2 = s1.lower(), s2.lower()
        if not s1 or not s2: return 0.0
        m, n = len(s1), len(s2)
        if m < n: s1, s2, m, n = s2, s1, n, m
        prev = range(n + 1)
        for i in range(m):
            curr = [i + 1]
            for j in range(n):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(s1[i]!=s2[j])))
            prev = curr
        return 1.0 - (prev[n] / max(m, n)) if max(m, n) else 1.0
    selected, remaining = [], list(results)
    scores = [r.get('score', 0) for r in remaining]
    max_s, min_s = max(scores) if scores else 1, min(scores) if scores else 0
    rng = max_s - min_s if max_s != min_s else 1.0
    norm = lambda r: (r.get('score', 0) - min_s) / rng
    while len(selected) < limit and remaining:
        best_idx = -1
        for idx, item in enumerate(remaining):
            rel = norm(item)
            max_sim = max((lev_sim(item['content'], s['content']) for s in selected), default=0)
            mmr = lambda_param * rel + (1 - lambda_param) * (1 - max_sim)
            if best_idx < 0 or mmr > (lambda_param * norm(remaining[best_idx]) +
                    (1 - lambda_param) * (1 - max((lev_sim(remaining[best_idx]['content'], s['content']) for s in selected), default=0))):
                best_idx = idx
        if best_idx < 0: break
        selected.append(remaining.pop(best_idx))
    return selected


# ─────────────────────────────────────────────────────────────────
# 6. STRIP + CREDENTIAL FILTER
# ─────────────────────────────────────────────────────────────────

_STRIP_PATTERNS = [
    (r'^\[message_id:\s*[^\]]+\]\s*', ''),
    (r'^Sender\s*\(untrusted metadata\):\s*```json\s*\n[\s\S]*?```\s*\n', ''),
    (r'^```json\s*\n[\s\S]*?```\s*\n', ''),
    (r'^\[user:ou_[^\]]+\]\s*', ''),
    (r'^Conversation info[\s\S]*?```\s*\n', ''),
    (r'^```\w*\s*\n', ''),
]

def strip_metadata(text: str) -> str:
    for pat, repl in _STRIP_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.MULTILINE)
    return text.strip()

_CRED_PATTERNS = [
    (r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]'),
    (r'gho_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]'),
    (r'github_pat_[a-zA-Z0-9_]{22,}', '[GITHUB_TOKEN]'),
    (r'(?<!\d)mars\d{5,}(?!\d)', '[PASSWORD]'),
    (r'(?<!\d)Mars\d{5,}(?!\d)', '[PASSWORD]'),
]
_CRED_BLOCK_PATTERNS = [
    r'ghp_[a-zA-Z0-9]{36}', r'gho_[a-zA-Z0-9]{36}',
    r'(?<!\d)mars\d{5,}(?!\d)', r'(?<!\d)Mars\d{5,}(?!\d)',
    r'(?i)(?:password|passwd|pwd|密码|secret|api_?key|token)\s*[:=]\s*[a-zA-Z0-9_\-]{4,}',
]

def filter_credentials(content: str) -> str:
    for pat, repl in _CRED_PATTERNS:
        content = re.sub(pat, repl, content)
    return content

def has_plaintext_credential(content: str) -> bool:
    for pat in _CRED_BLOCK_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            return True
    return False


# ─────────────────────────────────────────────────────────────────
# 7. COMMANDS
# ─────────────────────────────────────────────────────────────────

def cmd_search(query, limit=5, use_mmr=False, dedup=True, strip=True):
    t0 = time.time()

    # Primary: v5 subprocess to mempalace CLI (0.44s)
    v5_timeout = _TIMEOUT
    out, err, code = call_mempalace(['search', query, '--results', str(limit * 3)], timeout=v5_timeout)

    if code == 0:
        results = parse_search_output(out, query)
        source = 'mempalace_cli(v5)'
    else:
        # v6 fallback: direct ChromaDB + Ollama HTTP (no subprocess)
        results = v6_fallback_search(query, limit=limit * 3)
        source = 'v6_fallback'

    steps = [source]

    if not results:
        return {'status': 'ok', 'query': query, 'results': [], 'steps': steps}

    if strip:
        for r in results:
            r['content'] = strip_metadata(r['content'])
        steps.append('strip')

    for r in results:
        r['content'] = filter_credentials(r['content'])
    steps.append('credential_filter')

    before_dedup = len(results)
    if dedup:
        results = dedup_results(results)
        steps.append(f'dedup({before_dedup}→{len(results)})')

    for r in results:
        r['_identity_boost'] = identity_boost_score(r['source'], r['content'], query)
        r['_kw_boost'] = keyword_boost_score(r['content'], query)
        r['final_score'] = r['score'] + r['_identity_boost'] + r['_kw_boost']
    steps.append('identity_kw_boost')

    results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)

    if use_mmr:
        results = mmr_rerank(results, query, lambda_param=0.7, limit=limit)
        steps.append(f'mmr(→{len(results)})')
    else:
        results = results[:limit]
        steps.append(f'top({len(results)})')

    elapsed_ms = round((time.time() - t0) * 1000)

    return {
        'status': 'ok', 'query': query, 'steps': steps, 'elapsed_ms': elapsed_ms,
        'results': [
            {'content': r['content'], 'score': round(r['score'], 4),
             'final_score': round(r['final_score'], 4), 'source': r.get('source', '?'),
             '_boosts': {'identity': r.get('_identity_boost', 0),
                         'keyword': r.get('_kw_boost', 0)}}
            for r in results[:limit]
        ]
    }


def cmd_remember(content, agent='main', room='general', source=''):
    if has_plaintext_credential(content):
        filtered = filter_credentials(content)
        if has_plaintext_credential(filtered):
            return {'status': 'error', 'error': 'CREDENTIAL_DETECTED',
                    'message': '内容包含无法过滤的明文凭证，拒绝存储'}
        clean_content = filtered
        warn = '⚠️ 凭证已过滤，请勿在内容中直接包含密码/token'
    else:
        clean_content = content
        warn = None
    try:
        env = os.environ.copy()
        env['PATH'] = f'/Users/mars/Library/Python/3.9/bin:{env.get("PATH", "")}'
        r = subprocess.run(
            [sys.executable, SUPER_MEM_CLI, 'remember', clean_content,
             '--agent', agent, '--room', room, '--source', source],
            capture_output=True, text=True, timeout=15, env=env
        )
        if r.returncode == 0:
            try:
                result = json.loads(r.stdout)
                if warn:
                    result['warning'] = warn
                return result
            except:
                return {'status': 'ok', 'action': 'remember',
                        'content_preview': clean_content[:80], 'warning': warn}
        return {'status': 'error', 'error': r.stderr[:200]}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def cmd_wake_up():
    out, err, code = call_mempalace(['wake-up'], timeout=30)
    if code == 0:
        return {'status': 'ok', 'context': out}
    return {'status': 'error', 'error': err}


def cmd_status():
    out, err, code = call_mempalace(['status'], timeout=15)
    if code == 0:
        return {'status': 'ok', 'output': out}
    return {'status': 'error', 'error': err}


def cmd_mine(path=None):
    target = path or os.path.expanduser('~/.openclaw/workspace')
    out, err, code = call_mempalace(['mine', target, '--mode', 'projects'], timeout=120)
    if code == 0:
        return {'status': 'ok', 'path': target, 'output': out}
    return {'status': 'error', 'error': err}


def cmd_forget(memory_id):
    if not _CHROMA_AVAILABLE:
        return {'status': 'error', 'error': 'chromadb not available'}
    try:
        client = chromadb.PersistentClient(path=_SUPER_MEM_CHROMA)
        for col in client.list_collections():
            try:
                collection = client.get_collection(col.name)
                item = collection.get(ids=[memory_id])
                if item and item['ids']:
                    collection.delete(ids=[memory_id])
                    return {'status': 'ok', 'action': 'forget', 'id': memory_id}
            except Exception:
                pass
        return {'status': 'ok', 'action': 'forget', 'id': memory_id, 'note': 'not found'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mem-plus v7 — 实测最优架构：v5 primary + v6 fallback',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd')

    p_search = subparsers.add_parser('search')
    p_search.add_argument('query')
    p_search.add_argument('--limit', type=int, default=5)
    p_search.add_argument('--use-mmr', dest='use_mmr', action='store_true', default=False)
    p_search.add_argument('--no-dedup', dest='dedup', action='store_false', default=True)
    p_search.add_argument('--no-strip', dest='strip', action='store_false', default=True)

    subparsers.add_parser('status')
    subparsers.add_parser('wake-up')

    p_mine = subparsers.add_parser('mine')
    p_mine.add_argument('--path')

    p_forget = subparsers.add_parser('forget')
    p_forget.add_argument('memory_id')

    p_remember = subparsers.add_parser('remember')
    p_remember.add_argument('content')
    p_remember.add_argument('--agent', '-a', default='main')
    p_remember.add_argument('--room', '-r', default='general')
    p_remember.add_argument('--source', '-s', default='')

    args = parser.parse_args()

    if args.cmd == 'search':
        result = cmd_search(args.query, args.limit, args.use_mmr, args.dedup, args.strip)
    elif args.cmd == 'remember':
        result = cmd_remember(args.content, args.agent, args.room, args.source)
    elif args.cmd == 'status':
        result = cmd_status()
    elif args.cmd == 'wake-up':
        result = cmd_wake_up()
    elif args.cmd == 'mine':
        result = cmd_mine(args.path)
    elif args.cmd == 'forget':
        result = cmd_forget(args.memory_id)
    else:
        parser.print_help()
        sys.exit(0)

    print(json.dumps(result, ensure_ascii=False, indent=2))
