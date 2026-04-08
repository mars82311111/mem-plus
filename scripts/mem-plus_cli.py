#!/usr/bin/env python3
"""
mem-plus v8 — SuperMem 精华合并版
===================================
从 SuperMem v7 提炼合并的精华功能：
  1. exact_boost: 查询词精确出现在内容中 → +1.0 boost
  2. temporal_decay: 时间衰减（权重0.1，不干扰identity）
  3. filename detection: 查询像文件名时boost对应文档
  4. ngram_jaccard n=3: trigram去重（比bigram更准）

架构不变（v7实测最优）：
  - v5 subprocess to mempalace CLI (primary)
  - v6 direct fallback (timeout > 3s)
  - identity_boost + keyword_boost (from v7)
  - credential_filter (from v5)
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
# 3. PARSE — mempalace CLI markdown → structured results
# ─────────────────────────────────────────────────────────────────

def parse_search_output(output: str, query: str = '') -> list:
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
# 4. IDENTITY PRIORITY + KEYWORD BOOST (from v7)
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
    """Chinese bigram keyword boost (from v7)."""
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
# 5. SUPERMEM 精华功能 (NEW in v8)
# ─────────────────────────────────────────────────────────────────

def exact_boost_score(content: str, query: str) -> float:
    """
    SuperMem: if query appears verbatim in content, give +1.0 boost.
    If all query terms appear, give +0.8.
    Partial match: proportional boost.
    """
    q_lower = query.lower()
    c_lower = content.lower()
    if q_lower in c_lower:
        return 1.0
    terms = [t for t in query.split() if len(t) >= 2]
    if not terms:
        return 0.0
    matched = sum(1 for t in terms if t.lower() in c_lower)
    if matched == len(terms):
        return 0.8
    return 0.3 * (matched / len(terms))


def filename_detection(query: str, content: str) -> float:
    """
    SuperMem: if query looks like a filename (short, no spaces, has capitals),
    and content contains it as a markdown heading, boost.
    """
    if len(query) > 50 or ' ' in query.strip():
        return 0.0
    if not any(c.isupper() for c in query):
        return 0.0
    if re.search(rf'(?i)#\s*{re.escape(query)}\b', content):
        return 2.0
    return 0.0


def temporal_decay(mtime: float, half_life: int = 30) -> float:
    """
    SuperMem: time decay. Newer documents get higher weight.
    half_life=30 days: after 30 days, score *= 0.5
    """
    if not mtime or mtime <= 0:
        return 0.5
    days = (time.time() - float(mtime)) / 86400
    return max(0.1, min(1.0, 0.5 ** (days / half_life)))


def get_mtime_from_meta(meta: dict) -> float:
    """Extract mtime from metadata, trying multiple field names."""
    for field in ['filed_at', 'mtime', 'modified', 'updated', 'stored_at']:
        val = meta.get(field)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """SuperMem: n-gram Jaccard similarity."""
    def ngrams(s, n):
        s = s.lower()
        return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    a = ngrams(s1, n)
    b = ngrams(s2, n)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ─────────────────────────────────────────────────────────────────
# 6. DEDUP — ngram Jaccard (from SuperMem)
# ─────────────────────────────────────────────────────────────────

def dedup_results(results, threshold=0.85):
    if not results:
        return []
    # Step 1: keep highest-score per source
    by_source = {}
    for r in results:
        src = r.get('source', '')
        if src not in by_source or r.get('score', 0) > by_source[src].get('score', 0):
            by_source[src] = r
    # Step 2: n-gram Jaccard dedup
    deduped = []
    for r in by_source.values():
        if not any(ngram_jaccard(r['content'], e['content'], n=3) > threshold for e in deduped):
            deduped.append(r)
    return deduped


# ─────────────────────────────────────────────────────────────────
# 7. MMR (optional)
# ─────────────────────────────────────────────────────────────────

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
# 8. STRIP + CREDENTIAL FILTER
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
# 9. COMMANDS
# ─────────────────────────────────────────────────────────────────

def cmd_search(query, limit=5, use_mmr=False, dedup=True, strip=True):
    t0 = time.time()

    # Primary: v5 subprocess to mempalace CLI
    out, err, code = call_mempalace(['search', query, '--results', str(limit * 3)], timeout=_TIMEOUT)
    if code == 0:
        results = parse_search_output(out, query)
        source = 'mem-plus_cli(v8)'
    else:
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

    # All boosts
    TW = 0.1  # temporal weight: only 10%, won't override identity boost
    for r in results:
        r['_identity_boost'] = identity_boost_score(r['source'], r['content'], query)
        r['_kw_boost'] = keyword_boost_score(r['content'], query)
        r['_exact_boost'] = exact_boost_score(r['content'], query)
        r['_filename_boost'] = filename_detection(query, r['content'])
        mtime = get_mtime_from_meta(r.get('meta', {}))
        r['_temporal_decay'] = temporal_decay(mtime, half_life=30)
        decay = r['_temporal_decay']
        r['final_score'] = (
            r['score'] * (1 - TW) + r['score'] * decay * TW
            + r['_identity_boost']
            + r['_kw_boost']
            + r['_exact_boost']
            + r['_filename_boost']
        )
    steps.append('boosts(exact+temporal+identity+kw+filename)')

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
            {
                'content': r['content'],
                'score': round(r['score'], 4),
                'final_score': round(r['final_score'], 4),
                'source': r.get('source', '?'),
                '_boosts': {
                    'identity': r.get('_identity_boost', 0),
                    'keyword': r.get('_kw_boost', 0),
                    'exact': r.get('_exact_boost', 0),
                    'filename': r.get('_filename_boost', 0),
                    'temporal': round(r.get('_temporal_decay', 1), 3)
                }
            }
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
    # remember: store directly in SuperMem ChromaDB via direct Python
    import uuid
    if _CHROMA_AVAILABLE:
        try:
            client = chromadb.PersistentClient(path=_SUPER_MEM_CHROMA)
            col = client.get_or_create_collection("super_mem_shared")
            emb = ollama_embed_http([clean_content])[0]
            mid = f"mem_{uuid.uuid4().hex[:12]}"
            col.add(
                ids=[mid],
                embeddings=[emb],
                documents=[clean_content],
                metadatas=[{
                    "source_file": source or 'cli',
                    "room": room,
                    "agent": agent,
                    "stored_at": str(time.time()),
                    "mtime": str(time.time())
                }]
            )
            return {'status': 'ok', 'action': 'remember', 'id': mid, 'warning': warn}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    return {'status': 'error', 'error': 'chromadb not available'}


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
        description='mem-plus v8 — SuperMem 精华合并版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
SuperMem 合并功能:
  exact_boost: 查询词精确匹配 +1.0
  temporal_decay: 时间衰减（权重0.1）
  filename detection: 文件名查询 +2.0
  ngram_jaccard n=3: trigram 去重
'''
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
