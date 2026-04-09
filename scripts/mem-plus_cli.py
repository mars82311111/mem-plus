#!/usr/bin/env python3
"""
mem-plus v11 — 多业务线 Domain 架构
===================================
v11 新功能（3-tier + Domain Ownership）：
  三层记忆：Global Shared → Domain Shared → Private
  search --domain <name>: 搜索特定业务线
  remember --domain --agent: 写入对应域
  promote: 晋升知识（Private → Domain → Global）
  list-domains: 查看所有业务线

架构：
  - mempalace CLI (Global Shared primary)
  - ChromaDB (Domain/Agent private storage)
  - identity_boost + keyword_boost + exact_boost
  - 晋升机制：Private → Domain Shared → Global Shared
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

_MEM_CHROMA = os.path.expanduser("~/.openclaw/memory/chroma")

# ─────────────────────────────────────────────────────────────────
# DOMAIN CONFIG — 3-tier + Domain Ownership
# ─────────────────────────────────────────────────────────────────
_DOMAIN_CONFIG = os.path.expanduser("~/.openclaw/workspace/skills/mem-plus/domains/domains.json")


def _load_domains():
    if os.path.exists(_DOMAIN_CONFIG):
        try:
            with open(_DOMAIN_CONFIG) as f:
                return json.load(f)
        except Exception:
            pass
    return {"domains": {}, "agents": {"main": {"role": "CEO", "domain": None, "level": "global"}}, "version": "1.0"}


def _get_collection_name(domain: str = None, agent: str = None, scope: str = "shared") -> str:
    """Get ChromaDB collection name from domain/agent/scope.
    
    Priority: domain (if set) > agent (if set) > global
    - domain + agent + private → domain_{domain}_{agent}_private
    - domain + shared → domain_{domain}_shared
    - agent + private (no domain) → agent_{agent}_private
    - neither → global_shared
    """
    import re
    def _safe(s):
        """Remove illegal chars for ChromaDB collection names."""
        return re.sub(r'[^a-zA-Z0-9._-]', '_', str(s))
    # Agent without domain → agent_{agent}_private (higher priority than domain)
    if agent and not domain:
        return f"agent_{_safe(agent)}_private"
    # Domain with agent (private) → domain_{domain}_{agent}_private
    if domain:
        if scope == "private" and agent:
            return f"domain_{_safe(domain)}_{_safe(agent)}_private"
        return f"domain_{_safe(domain)}_shared"
    # Agent with domain uses domain path above; this is fallback
    if agent:
        return f"agent_{_safe(agent)}_private"
    return "global_shared"


def _search_chroma(query_text: str, collection_name: str, limit: int = 5) -> list:
    """Search a specific ChromaDB collection and its versioned variants.
    
    Searches: collection_name, collection_name_v2, collection_name_v3...
    This handles embedding model upgrades (e.g. nomic 768-dim → bge-m3 1024-dim)
    where old and new data live in separate collections.
    """
    if not _CHROMA_AVAILABLE:
        return []
    q_emb = ollama_embed_http([query_text])[0]
    try:
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
        all_items = []
        # Search base collection + all vN variants
        for variant in [collection_name] + [f"{collection_name}_v{n}" for n in range(2, 10)]:
            try:
                col = client.get_collection(variant)
                raw = col.query(
                    query_embeddings=[q_emb],
                    n_results=limit * 2,
                    include=["documents", "metadatas"]
                )
                docs = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas", [[]])[0]
                if not docs:
                    continue
                for i, doc in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    sf = meta.get("source_file", "")
                    source = os.path.basename(sf) if sf else sf
                    all_items.append({
                        "content": doc, "source": source,
                        "score": 1.0 - (i / max(len(docs), 1)),
                        "meta": meta, "collection": variant
                    })
            except Exception:
                break  # Collection doesn't exist, stop searching variants
        return all_items
    except Exception:
        return []


def ollama_embed_http(texts: list) -> list:
    import urllib.request
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": "bge-m3", "prompt": text}).encode()
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
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
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


# ─────────────────────────────────────────────────────────────────
# 4b. CHINESE SYNONYM EXPANSION (P2 Fix: Chinese semantic search)
# ─────────────────────────────────────────────────────────────────

# 常见中文类别词 → 同类词扩展（解决 nomic-embed-text 对中文语义理解不足）
# 当 query 包含类别词时，扩展匹配同类具体物品/概念
_CHINESE_CATEGORY_EXPANSION = {
    '水果': ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '草莓', '桃子', '梨', '芒果', '菠萝'],
    '水果类': ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '草莓', '桃子', '梨', '芒果', '菠萝'],
    '水果的': ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '草莓', '桃子', '梨', '芒果', '菠萝'],
    '电脑': ['笔记本', '台式机', 'PC', '计算机', 'Mac', 'Windows'],
    '计算机': ['电脑', '笔记本', '台式机', 'PC', 'Mac'],
    '电影': ['影片', '片子', '剧情', '导演', '演员', '豆瓣'],
    '音乐': ['歌曲', '歌', '歌手', '专辑', '歌词', 'MP3'],
    '书': ['书籍', '图书', '阅读', '作者', '出版社', 'Kindle'],
    '书籍': ['书', '图书', '阅读', '作者', '出版社'],
    '网站': ['网页', 'URL', '链接', '域名', 'Web'],
    '应用': ['APP', '软件', '工具', '程序', 'App'],
    '手机': ['iPhone', '安卓', 'Android', 'APP', '微信'],
}

# 双向同义词
_CHINESE_BIDIRECTIONAL = {
    'AI': ['人工智能', '机器学习', '深度学习'],
    '人工智能': ['AI', '机器学习', '深度学习'],
    '机器学习': ['ML', 'AI', '人工智能'],
    '深度学习': ['DL', 'AI', '神经网络'],
    'LLM': ['大模型', '语言模型', 'GPT'],
    '大模型': ['LLM', '语言模型', 'GPT', 'ChatGPT'],
}


def expand_chinese_query(query: str) -> set:
    """Expand Chinese query with category synonyms.
    
    When query contains a category word (e.g. '水果'),
    expand it to include specific items in that category.
    This compensates for nomic-embed-text's limited Chinese understanding.
    """
    expanded = set()
    q_lower = query.lower()
    
    # Direct category expansion
    for cat, items in _CHINESE_CATEGORY_EXPANSION.items():
        if cat in q_lower or cat in query:
            expanded.update(items)
    
    # Bidirectional synonym expansion
    for word, synonyms in _CHINESE_BIDIRECTIONAL.items():
        if word in q_lower or word in query:
            expanded.update(synonyms)
    
    return expanded


def chinese_concept_match_score(content: str, query: str) -> float:
    """Check if content contains expanded query concepts.
    
    Returns a boost score based on how many expanded terms
    from the query appear in the content.
    """
    if not is_chinese(query):
        return 0.0
    
    expanded = expand_chinese_query(query)
    if not expanded:
        return 0.0
    
    content_lower = content.lower()
    matches = sum(1 for term in expanded if term.lower() in content_lower)
    
    if matches == 0:
        return 0.0
    
    # Score: up to +6.0 based on number of matches (1.0 per matched term)
    # This is a strong boost to compensate for nomic-embed-text Chinese weakness
    return min(6.0, matches * 1.0)


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
    SuperMem exact_boost: if query appears verbatim in content, give +2.0 boost.
    If all query terms appear, give +1.5.
    Partial match: proportional boost up to +0.5.
    """
    q_lower = query.lower()
    c_lower = content.lower()
    if q_lower in c_lower:
        return 2.0
    terms = [t for t in query.split() if len(t) >= 2]
    if not terms:
        return 0.0
    matched = sum(1 for t in terms if t.lower() in c_lower)
    if matched == len(terms):
        return 1.5
    return 0.5 * (matched / len(terms))


def _looks_like_filename(query: str) -> bool:
    """判断查询是否像文件名。"""
    if len(query) > 50 or len(query) < 3:
        return False
    if ' ' in query.strip():
        return False
    if not any(c.isupper() for c in query):
        return False
    return True


def _filename_direct_inject(query: str, results: list) -> list:
    """
    根因修复：MemPalace 向量搜索对文件名 heading 查询从根本上失效。
    直接读文件系统，精确查找包含 "# {query}" 的文件。
    找到后：若已存在结果中→提升为 rank1（置顶+超高boost）；
           若不存在→注入到 rank1。
    """
    if not _looks_like_filename(query):
        return results

    workspace = os.path.expanduser('~/.openclaw/workspace')

    # 直接扫描 workspace 找 heading 匹配的文件
    matched_result = None
    try:
        for fname in os.listdir(workspace):
            fpath = os.path.join(workspace, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.md', '.txt', '.json', '.py', '.yaml', '.yml'):
                continue
            try:
                with open(fpath, encoding='utf-8', errors='ignore') as f:
                    first_lines = ''.join(f.readlines()[:50])
                # heading 精确匹配：行首 # 后跟 query（忽略大小写）
                heading_pattern = rf'^#\s*{re.escape(query)}\b'
                if re.search(heading_pattern, first_lines, re.MULTILINE | re.IGNORECASE):
                    with open(fpath, encoding='utf-8', errors='ignore') as f:
                        full_content = f.read(50000)
                    matched_result = {
                        'content': full_content,
                        'score': 999.0,
                        'source': fname,
                        'meta': {'mtime': os.path.getmtime(fpath)},
                        '_injected': True,
                        '_filename_boost': 999.0,
                    }
                    break
            except Exception:
                continue
    except Exception:
        pass

    if not matched_result:
        return results

    # 从结果中移除同名文件（如果有）
    results = [r for r in results if r.get('source', '') != matched_result['source']]
    # 插入到 rank1
    results.insert(0, matched_result)
    return results


def filename_detection(query: str, content: str) -> float:
    """
    SuperMem: if query looks like a filename (short, no spaces, has capitals),
    and content contains it as a markdown heading, boost.
    Note: _filename_direct_inject handles injection; this is secondary boost.
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
    """
    N-gram Jaccard deduplication (from SuperMem v7).
    Injected results (_injected=True) are protected from dedup — they are
    filesystem-precise matches (e.g. filename heading lookup) and must not be removed.
    """
    if not results:
        return []
    # Step 1: keep highest-score per source
    by_source = {}
    for r in results:
        src = r.get('source', '')
        if src not in by_source or r.get('score', 0) > by_source[src].get('score', 0):
            by_source[src] = r
    # Step 2: n-gram Jaccard dedup — injected results are protected
    deduped = []
    for r in by_source.values():
        # Injected results (filesystem-precise matches) skip dedup check
        if r.get('_injected'):
            deduped.append(r)
            continue
        if not any(ngram_jaccard(r['content'], e['content'], n=3) > threshold for e in deduped):
            deduped.append(r)
    return deduped


# ─────────────────────────────────────────────────────────────────
# 7. MMR (optional)
# ─────────────────────────────────────────────────────────────────

def mmr_rerank(results, query, lambda_param=0.7, limit=5):
    """
    MMR reranking using ngram Jaccard diversity (from SuperMem v7).
    Replaces slow Levenshtein O(n*m) with fast ngram Jaccard O(n).
    For long docs (MEMORY.md 5000+ chars), this is 10-100x faster.
    """
    if not results or len(results) <= limit:
        return results
    selected, remaining = [], list(results)
    scores = [r.get('score', 0) for r in remaining]
    max_s, min_s = max(scores) if scores else 1, min(scores) if scores else 0
    rng = max_s - min_s if max_s != min_s else 1.0
    norm = lambda r: (r.get('score', 0) - min_s) / rng
    while len(selected) < limit and remaining:
        best_idx = -1
        best_mmr = -float('inf')
        for idx, item in enumerate(remaining):
            rel = norm(item)
            # Use ngram_jaccard instead of Levenshtein — O(n) not O(n*m)
            max_sim = max((ngram_jaccard(item['content'], s['content'], n=3) for s in selected), default=0)
            mmr = lambda_param * rel + (1 - lambda_param) * (1 - max_sim)
            if mmr > best_mmr:
                best_mmr = mmr
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

def cmd_search(query, limit=5, use_mmr=False, dedup=True, strip=True, tw=0.1, hl=30, wing=None, room=None, domain=None, agent=None):
    t0 = time.time()
    all_results = []
    sources = []

    # 1. Mempalace CLI — Global Shared (always searched)
    mp_args = ['search', query, '--results', str(limit * 3)]
    if wing:
        mp_args.extend(['--wing', wing])
    if room:
        mp_args.extend(['--room', room])
    out, err, code = call_mempalace(mp_args, timeout=_TIMEOUT)
    if code == 0:
        mp_results = parse_search_output(out, query)
        all_results.extend([{**r, '_source_layer': 'global'} for r in mp_results])
        sources.append('mempalace(global)')
    else:
        sources.append('mempalace(failed)')

    # 2. Domain Shared — if domain specified
    if domain:
        coll = _get_collection_name(domain=domain, scope='shared')
        domain_results = _search_chroma(query, coll, limit=limit)
        all_results.extend([{**r, '_source_layer': f'domain:{domain}'} for r in domain_results])
        sources.append(f'domain:{domain}_shared')

    # 3. Agent Private — if agent specified
    if agent:
        coll = _get_collection_name(domain=domain, agent=agent, scope='private')
        priv_results = _search_chroma(query, coll, limit=limit)
        all_results.extend([{**r, '_source_layer': f'private:{agent}'} for r in priv_results])
        sources.append(f'agent:{agent}_private')

    if not all_results:
        return {'status': 'ok', 'query': query, 'results': [], 'steps': sources}

    results = all_results
    steps = sources

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

    # ═══════════════════════════════════════════════════════
    # ROOT CAUSE FIX: filename direct inject
    # 当查询像文件名时（query="SOUL.md"），MemPalace 向量搜索
    # 根本不返回 SOUL.md，因为向量空间不匹配。
    # 这里用 heading 搜索直接召回并注入 results 最前。
    # ═══════════════════════════════════════════════════════
    results = _filename_direct_inject(query, results)

    if dedup:
        # dedup_results now protects _injected results (SuperMem v7 pattern)
        results = dedup_results(results)
        steps.append(f'dedup({before_dedup}→{len(results)})')

    # All boosts
    # TW=0.3 (SuperMem v7 optimum): temporal decay gets 30% weight, balanced with vector score
    for r in results:
        # 注入结果（_injected=True）已预置 filename_boost=999.0，跳过重算
        if r.get('_injected'):
            r['_identity_boost'] = identity_boost_score(r['source'], r['content'], query)
            r['_kw_boost'] = 0.0
            r['_exact_boost'] = 2.0  # heading 精确匹配
            r['_filename_boost'] = r.get('_filename_boost', 999.0)
            r['_chinese_concept_boost'] = 0.0
        else:
            r['_identity_boost'] = identity_boost_score(r['source'], r['content'], query)
            r['_kw_boost'] = keyword_boost_score(r['content'], query)
            r['_exact_boost'] = exact_boost_score(r['content'], query)
            r['_filename_boost'] = filename_detection(query, r['content'])
            # P2 Fix: Chinese concept matching (category → specific items)
            r['_chinese_concept_boost'] = chinese_concept_match_score(r['content'], query)
        mtime = get_mtime_from_meta(r.get('meta', {}))
        r['_temporal_decay'] = temporal_decay(mtime, half_life=hl)
        decay = r['_temporal_decay']
        r['final_score'] = (
            r['score'] * (1 - tw) + r['score'] * decay * tw
            + r['_identity_boost']
            + r['_kw_boost']
            + r['_exact_boost']
            + r['_filename_boost']
            + r['_chinese_concept_boost']
        )
    steps.append('boosts(exact+temporal+identity+kw+filename+cn_concept)')

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
                    'temporal': round(r.get('_temporal_decay', 1), 3),
                    'cn_concept': round(r.get('_chinese_concept_boost', 0), 3)
                }
            }
            for r in results[:limit]
        ]
    }


def cmd_remember(content, agent='main', room='general', source='', domain=None):
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
    # remember: store in domain/agent private ChromaDB collection
    # FIX v1: Use content hash as ID → identical content = identical ID → write-time dedup
    # FIX v2: Handle embedding dimension mismatch (e.g. bge-m3 1024-dim vs old 768-dim)
    import hashlib
    if _CHROMA_AVAILABLE:
        try:
            client = chromadb.PersistentClient(path=_MEM_CHROMA)
            coll_name = _get_collection_name(domain=domain, agent=agent, scope='private')
            emb = ollama_embed_http([clean_content])[0]
            new_dim = len(emb)
            
            # Content hash as ID: identical content → identical ID → ChromaDB upsert = dedup
            content_hash = hashlib.sha256(clean_content.encode()).hexdigest()[:16]
            mid = f"mem_{content_hash}"
            
            # Try the original collection first
            def try_write(target_coll_name):
                col = client.get_or_create_collection(target_coll_name)
                # Check if identical content already exists
                existing = col.get(ids=[mid], include=[])
                if existing['ids']:
                    return {'status': 'ok', 'action': 'remember', 'id': mid, 'warning': warn, 'note': 'already exists, skipped'}
                col.add(
                    ids=[mid],
                    embeddings=[emb],
                    documents=[clean_content],
                    metadatas=[{
                        "source_file": source or 'cli',
                        "room": room,
                        "agent": agent,
                        "stored_at": str(time.time()),
                        "mtime": str(time.time()),
                        "embedding_dim": new_dim
                    }]
                )
                return {'status': 'ok', 'action': 'remember', 'id': mid, 'warning': warn}
            
            try:
                return try_write(coll_name)
            except Exception as write_err:
                err_str = str(write_err)
                # ChromaDB error for dimension mismatch: "Expected 768-dim, got 1024-dim"
                if 'dimension' in err_str.lower() or 'dimension' in err_str:
                    # Collection has old data with different dimension → create versioned collection
                    v2_coll_name = f"{coll_name}_v2"
                    try:
                        result = try_write(v2_coll_name)
                        result['note'] = (result.get('note','') + f' | NOTE: created new v2 collection ({new_dim}-dim) due to dimension mismatch with existing data').strip()
                        return result
                    except Exception:
                        pass  # Fall through to error
                return {'status': 'error', 'error': str(write_err)}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    return {'status': 'error', 'error': 'chromadb not available'}


def cmd_wake_up(wing=None):
    args = ['wake-up']
    if wing:
        args.extend(['--wing', wing])
    out, err, code = call_mempalace(args, timeout=30)
    if code == 0:
        return {'status': 'ok', 'context': out, 'wing': wing}
    return {'status': 'error', 'error': err}


def cmd_status():
    out, err, code = call_mempalace(['status'], timeout=15)
    if code == 0:
        return {'status': 'ok', 'output': out}
    return {'status': 'error', 'error': err}


def cmd_list_agents():
    """List all agents that have memory collections."""
    if not _CHROMA_AVAILABLE:
        return {'status': 'error', 'error': 'chromadb not available'}
    try:
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
        cols = client.list_collections()
        agents = sorted(
            c.name.replace('agent_', '').replace('_private', '')
            for c in cols
            if c.name.startswith('agent_') and c.name != 'global_shared'
        )
        return {'status': 'ok', 'agents': agents, 'total': len(agents)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def _bridge_sync() -> dict:
    """Sync MemPalace drawers → SuperMem ChromaDB."""
    if not _CHROMA_AVAILABLE:
        return {'error': 'chromadb not available'}
    try:
        import chromadb
        mp = chromadb.PersistentClient(path=os.path.expanduser('~/.mempalace/palace'))
        mp_col = mp.get_collection('mempalace_drawers')
        items = mp_col.get(limit=10000, include=['documents', 'metadatas'])
        if not items['ids']:
            return {'synced': 0, 'note': 'MemPalace empty'}
        shared = chromadb.PersistentClient(path=_MEM_CHROMA)
        sc = shared.get_or_create_collection('super_mem_shared', metadata={'shared': 'true'})
        old_ids = [mid for mid in sc.get(limit=10000, include=[])['ids']
                   if mid.startswith('mp_') and not mid.startswith('mp_bridge_')]
        if old_ids:
            sc.delete(ids=old_ids)
        docs, metas, ids = [], [], []
        for i, did in enumerate(items['ids']):
            doc = items['documents'][i] if items['documents'] else ''
            meta = items['metadatas'][i] if items['metadatas'] else {}
            docs.append(filter_credentials(doc))
            metas.append({**meta, 'source': 'mem-plus_bridge', 'original_id': did})
            ids.append(f'mp_bridge_{did}')
        embs = ollama_embed_http(docs)
        sc.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        return {'synced': len(docs), 'deleted_old': len(old_ids)}
    except Exception as e:
        return {'error': str(e)}


def cmd_list_domains(verbose=False):
    """List all domains and their collections."""
    cfg = _load_domains()
    if not _CHROMA_AVAILABLE:
        return {'status': 'error', 'error': 'chromadb not available'}
    try:
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
        cols = {c.name: c for c in client.list_collections()}
        domains = list(cfg.get('domains', {}).keys())
        result = {'status': 'ok', 'domains': domains, 'collections': {}}
        for name in cols:
            if name.startswith('domain_') or name.startswith('agent_') or name == 'global_shared':
                count = cols[name].count()
                result['collections'][name] = count
        if verbose:
            result['config'] = cfg
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def cmd_promote(memory_id, from_domain, from_agent, to_global=False, to_domain=None):
    """Promote a memory: Private → Domain Shared → Global Shared.
    
    --to-global: promote to Global Shared (mempalace)
    --to-domain <name>: promote to Domain Shared (domain_<name>_shared)
    
    One of --to-global or --to-domain must be specified.
    """
    if not _CHROMA_AVAILABLE:
        return {'status': 'error', 'error': 'chromadb not available'}
    if to_global and to_domain:
        return {'status': 'error', 'error': 'cannot use --to-global and --to-domain at the same time'}
    if not to_global and not to_domain:
        return {'status': 'error', 'error': 'must specify --to-global or --to-domain <name>'}
    if to_domain == 'domain':
        return {'status': 'error', 'error': '--to-domain <name>: <name> must be a valid domain name (e.g. strategy, product), not the literal "domain"'}
    try:
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
        from_coll_name = _get_collection_name(domain=from_domain, agent=from_agent, scope='private')
        from_col = client.get_collection(from_coll_name)
        items = from_col.get(ids=[memory_id], include=['documents', 'metadatas'])
        if not items['ids']:
            return {'status': 'error', 'error': 'memory not found'}
        doc = items['documents'][0] if items['documents'] else ''
        meta = items['metadatas'][0] if items['metadatas'] else {}
        if to_global:
            out, err, code = call_mempalace(['remember', doc], timeout=30)
            if code != 0:
                return {'status': 'error', 'error': err}
            from_col.delete(ids=[memory_id])
            return {'status': 'ok', 'action': 'promoted', 'from': from_coll_name, 'to': 'mempalace(global)'}
        else:
            to_coll = _get_collection_name(domain=to_domain, scope='shared')
            to_col = client.get_or_create_collection(to_coll)
            emb = ollama_embed_http([doc])[0]
            to_col.add(ids=[f'promoted_{memory_id}'], embeddings=[emb],
                       documents=[doc], metadatas=[{**meta, 'promoted_from': from_coll_name}])
            from_col.delete(ids=[memory_id])
            return {'status': 'ok', 'action': 'promoted', 'from': from_coll_name, 'to': to_coll}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def cmd_forget(memory_id, domain=None, agent=None):
    """Delete a specific memory from domain/agent private collection."""
    if not _CHROMA_AVAILABLE:
        return {'status': 'error', 'error': 'chromadb not available'}
    try:
        client = chromadb.PersistentClient(path=_MEM_CHROMA)
        coll_name = _get_collection_name(domain=domain, agent=agent, scope='private')
        col = client.get_collection(coll_name)
        col.delete(ids=[memory_id])
        return {'status': 'ok', 'action': 'forget', 'id': memory_id, 'collection': coll_name}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def cmd_mine(path=None, do_bridge=True):
    target = path or os.path.expanduser('~/.openclaw/workspace')
    out, err, code = call_mempalace(['mine', target, '--mode', 'projects'], timeout=120)
    if code != 0:
        return {'status': 'error', 'error': err}
    result = {'status': 'ok', 'path': target, 'output': out}
    if do_bridge:
        sync = _bridge_sync()
        result['bridge_sync'] = sync
    return result




# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mem-plus v10 — MemPalace wing/room 集成版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''mem-plus v10 新功能:
  search --wing <project> --room <room>: 项目+房间双重过滤召回
  wake-up --wing <project>: 项目级上下文唤醒
  exact_boost: 全词2.0 / 所有词1.5 / 部分词0.5
  --tw --hl: 可调 temporal 参数
'''
    )
    subparsers = parser.add_subparsers(dest='cmd')

    p_search = subparsers.add_parser('search')
    p_search.add_argument('query')
    p_search.add_argument('--limit', type=int, default=5)
    p_search.add_argument('--use-mmr', dest='use_mmr', action='store_true', default=False)
    p_search.add_argument('--no-dedup', dest='dedup', action='store_false', default=True)
    p_search.add_argument('--no-strip', dest='strip', action='store_false', default=True)
    p_search.add_argument('--tw', type=float, default=0.3)  # 0.3: matches SuperMem v7 optimum
    p_search.add_argument('--hl', type=int, default=30)
    p_search.add_argument('--wing', default=None)
    p_search.add_argument('--room', default=None)
    p_search.add_argument('--domain', '-d', default=None)
    p_search.add_argument('--agent', '-a', default=None)

    subparsers.add_parser('status')

    p_wakeup = subparsers.add_parser('wake-up')
    p_wakeup.add_argument('--wing', default=None)

    p_list = subparsers.add_parser('list-domains')
    p_list.add_argument('--verbose', '-v', action='store_true')

    p_promote = subparsers.add_parser('promote')
    p_promote.add_argument('memory_id')
    p_promote.add_argument('--from-domain', required=True)
    p_promote.add_argument('--from-agent')
    p_promote.add_argument('--to-global', dest='to_global', action='store_true', default=False)
    p_promote.add_argument('--to-domain', dest='to_domain', default=None)

    subparsers.add_parser('list-agents')

    p_mine = subparsers.add_parser('mine')
    p_mine.add_argument('--path')
    p_mine.add_argument('--no-bridge', dest='bridge', action='store_false', default=True)

    p_forget = subparsers.add_parser('forget')
    p_forget.add_argument('memory_id')
    p_forget.add_argument('--domain')
    p_forget.add_argument('--agent')

    p_remember = subparsers.add_parser('remember')
    p_remember.add_argument('content')
    p_remember.add_argument('--agent', '-a', default='main')
    p_remember.add_argument('--room', '-r', default='general')
    p_remember.add_argument('--source', '-s', default='')
    p_remember.add_argument('--domain', '-d', default=None)

    args = parser.parse_args()

    if args.cmd == 'search':
        result = cmd_search(args.query, args.limit, args.use_mmr, args.dedup,
                            args.strip, tw=args.tw, hl=args.hl,
                            wing=args.wing, room=args.room,
                            domain=args.domain, agent=args.agent)
    elif args.cmd == 'wake-up':
        result = cmd_wake_up(wing=args.wing)
    elif args.cmd == 'status':
        result = cmd_status()
    elif args.cmd == 'mine':
        result = cmd_mine(args.path, do_bridge=args.bridge)
    elif args.cmd == 'list-agents':
        result = cmd_list_agents()
    elif args.cmd == 'list-domains':
        result = cmd_list_domains(verbose=args.verbose)
    elif args.cmd == 'promote':
        result = cmd_promote(args.memory_id, args.from_domain, args.from_agent,
                               to_global=args.to_global, to_domain=args.to_domain)
    elif args.cmd == 'remember':
        result = cmd_remember(args.content, args.agent, args.room, args.source, domain=args.domain)
    elif args.cmd == 'forget':
        result = cmd_forget(args.memory_id, domain=args.domain, agent=args.agent)
    else:
        parser.print_help()
        sys.exit(0)

    print(json.dumps(result, ensure_ascii=False, indent=2))
