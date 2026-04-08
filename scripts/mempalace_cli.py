#!/usr/bin/env python3
"""
MemPalace Enhanced CLI Wrapper v4
================================
Fixes from v3:
- parse_search_output: correctly splits individual [N] results
- Uses absolute score for proper ranking
- All MMR/dedup/strip actually work now
"""
import sys
import os
import json
import argparse
import re

MEMPALACE_CLI = '/Users/mars/Library/Python/3.9/bin/mempalace'
RERANKER = '/Users/mars/.openclaw/workspace/skills/mempalace-memory/scripts/mempalace_reranker.py'
WORKSPACE = os.path.expanduser('~/.openclaw/workspace')

def call_mempalace(args, timeout=30):
    """Call mempalace CLI directly."""
    import subprocess
    env = os.environ.copy()
    env['PATH'] = f'/Users/mars/Library/Python/3.9/bin:{env.get("PATH", "")}'
    result = subprocess.run(
        [MEMPALACE_CLI] + args,
        capture_output=True, text=True, timeout=timeout,
        env=env
    )
    return result.stdout, result.stderr, result.returncode


# ─────────────────────────────────────────────────────────────────
# 1. PARSE — FIXED: correctly split individual [N] results
# ─────────────────────────────────────────────────────────────────

def parse_search_output(output: str, query: str) -> list:
    """
    Parse mempalace CLI markdown output into structured results.
    
    Format:
      ============================================================
        Results for: "query"
      ============================================================
      
        [1] workspace / room
            Source: filename
            Match:  score
            
            content here
        ────────────────────────────────────────────────────────
        [2] workspace / room
            Source: filename
            Match:  score
            
            content here
    """
    results = []
    
    # Split by separator lines (5+ dashes on their own line, with optional whitespace)
    # Separator appears BETWEEN results
    blocks = re.split(r'\n\s*─{5,}\s*\n', output)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Case 1: block starts with [N] header — a normal result block
        if re.match(r'\[\d+\]', block):
            result = _extract_result(block)
            if result:
                results.append(result)
            continue
        
        # Case 2: header block containing first [1] result inline
        # (when there's no leading separator before [1])
        first_result_start = block.find('[1]')
        if first_result_start != -1:
            first_block = block[first_result_start:]
            result = _extract_result(first_block)
            if result:
                results.append(result)
    
    return results


def _extract_result(block: str) -> dict:
    """Extract source, score, and content from a result block."""
    header_match = re.match(r'\[(\d+)\]\s+(\S+)\s+/\s+(\S+)', block)
    if not header_match:
        return None
    
    source_match = re.search(r'Source:\s*(.+?)(?:\n|$)', block)
    match_score_match = re.search(r'Match:\s*([-\d.]+)', block)
    
    source = source_match.group(1).strip() if source_match else ''
    raw_score = float(match_score_match.group(1)) if match_score_match else 0.0
    
    # Content: after the blank line following Match line
    match_pos = block.find('Match:')
    if match_pos == -1:
        return {
            'content': block[header_match.end():].strip(),
            'score': abs(raw_score),
            'source': source,
            'match_score': raw_score
        }
    
    line_end = block.find('\n', match_pos)
    blank = block.find('\n\n', line_end)
    content = block[blank + 2:].strip() if blank != -1 else block[line_end + 1:].strip()
    
    return {
        'content': content,
        'score': raw_score,  # Use RAW score — positive = best match (mempalace pre-ranked)
        'source': source,
        'match_score': raw_score
    }


# ─────────────────────────────────────────────────────────────────
# 2. DEDUP — Levenshtein-based (from mempalace_reranker.py)
# ─────────────────────────────────────────────────────────────────

def dedup_results(results, threshold=0.85):
    """
    Levenshtein-based deduplication + same-source dedup.
    Keeps the highest-scoring result from each source file,
    then removes Levenshtein duplicates.
    """
    if not results:
        return []
    
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]
    
    def similarity(s1, s2):
        s1 = s1.lower(); s2 = s2.lower()
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (levenshtein(s1, s2) / max_len)
    
    # Step 1: keep highest-scoring result per source file
    by_source = {}
    for r in results:
        src = r.get('source', '')
        if src not in by_source or r.get('score', 0) > by_source[src].get('score', 0):
            by_source[src] = r
    
    # Step 2: Levenshtein dedup among unique sources
    deduped = []
    for r in by_source.values():
        content = r.get('content', '')
        is_dup = any(similarity(content, e.get('content', '')) > threshold for e in deduped)
        if not is_dup:
            deduped.append(r)
    
    return deduped


# ─────────────────────────────────────────────────────────────────
# 3. MMR — Maximum Marginal Relevance reranking
# ─────────────────────────────────────────────────────────────────

def mmr_rerank(results, query, lambda_param=0.7, limit=5):
    """
    Maximum Marginal Relevance reranking.
    Balances relevance (score) with diversity (maximize dissimilarity to already selected).
    """
    if not results:
        return []
    if len(results) <= limit:
        return results
    
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]
    
    def similarity(s1, s2):
        s1 = s1.lower(); s2 = s2.lower()
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (levenshtein(s1, s2) / max_len)
    
    selected = []
    remaining = list(results)
    
    # Normalize scores (raw score, higher is better — mempalace pre-ranked)
    max_s = max((r.get('score', 0) for r in remaining), default=1)
    min_s = min((r.get('score', 0) for r in remaining), default=0)
    score_range = max_s - min_s if max_s != min_s else 1.0
    
    def norm(r):
        return (r.get('score', 0) - min_s) / score_range
    
    while len(selected) < limit and remaining:
        best_score = -float('inf')
        best_item = None
        best_idx = -1
        
        for idx, item in enumerate(remaining):
            relevance = norm(item)
            max_sim = max(
                (similarity(item.get('content', ''), s.get('content', '')) for s in selected),
                default=0
            )
            diversity = 1.0 - max_sim
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
                best_idx = idx
        
        if best_item is not None:
            selected.append(best_item)
            remaining.pop(best_idx)
        else:
            break
    
    return selected


# ─────────────────────────────────────────────────────────────────
# 4. STRIP — remove OpenClaw metadata from injected content
# ─────────────────────────────────────────────────────────────────

STRIP_PATTERNS = [
    (r'^\[message_id:\s*[^\]]+\]\s*', ''),
    (r'^Sender\s*\(untrusted metadata\):\s*```json\s*\n[\s\S]*?```\s*\n', ''),
    (r'^```json\s*\n[\s\S]*?```\s*\n', ''),
    (r'^\[user:ou_[^\]]+\]\s*', ''),
    (r'^Conversation info[\s\S]*?```\s*\n', ''),
    (r'^```\w*\s*\n', ''),
]

def strip_metadata(text: str) -> str:
    for pat, repl in STRIP_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.MULTILINE)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# 5. COMMANDS
# ─────────────────────────────────────────────────────────────────

def cmd_search(query, limit=5, use_mmr=True, dedup=True, strip=True):
    """Enhanced search with MMR + dedup + strip."""
    out, err, code = call_mempalace(['search', query, '--results', str(limit * 3)])
    
    if code != 0:
        return {'status': 'error', 'error': err}
    
    # Parse individual [N] results
    results = parse_search_output(out, query)
    
    if not results:
        return {'status': 'ok', 'query': query, 'results': [], 'steps': ['parse_error']}
    
    steps = ['mempalace_native']
    
    # Strip metadata
    if strip:
        for r in results:
            r['content'] = strip_metadata(r['content'])
        steps.append('strip')
    
    # Deduplicate (source dedup + Levenshtein)
    before_dedup = len(results)
    if dedup:
        results = dedup_results(results)
        steps.append(f'dedup({before_dedup}→{len(results)})')
    
    # Sort by raw score descending (mempalace pre-ranked, raw score = correct ordering)
    results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    # MMR rerank
    before_mmr = len(results)
    if use_mmr:
        results = mmr_rerank(results, query, lambda_param=0.7, limit=limit)
        steps.append(f'mmr({before_mmr}→{len(results)})')
    else:
        results = results[:limit]
        steps.append('sorted')
    
    return {
        'status': 'ok',
        'query': query,
        'steps': steps,
        'results': [
            {
                'content': r['content'],
                'score': round(r['score'], 4),
                'source': r.get('source', '?'),
                'match_score': round(r.get('match_score', 0), 3)
            }
            for r in results[:limit]
        ]
    }


def cmd_wake_up():
    """Wake up with full context (L0 + L1 layers)."""
    out, err, code = call_mempalace(['wake-up'], timeout=30)
    if code == 0:
        return {'status': 'ok', 'context': out}
    return {'status': 'error', 'error': err}


def cmd_status():
    """Check mempalace health and index status."""
    out, err, code = call_mempalace(['status'], timeout=15)
    if code == 0:
        return {'status': 'ok', 'output': out}
    return {'status': 'error', 'error': err}


def cmd_mine(path=None):
    """Mine a directory for new memories."""
    target = path or WORKSPACE
    out, err, code = call_mempalace(['mine', target, '--mode', 'projects'], timeout=120)
    if code == 0:
        return {'status': 'ok', 'path': target, 'output': out}
    return {'status': 'error', 'error': err}


def cmd_forget(memory_id):
    """Delete a memory by ID from ChromaDB."""
    try:
        import chromadb
        palace_path = os.path.expanduser('~/.mempalace/palace')
        client = chromadb.PersistentClient(path=palace_path)
        collections = client.list_collections()
        deleted = False
        for col in collections:
            try:
                collection = client.get_collection(col.name)
                try:
                    item = collection.get(ids=[memory_id])
                    if item and item['ids']:
                        collection.delete(ids=[memory_id])
                        deleted = True
                        break
                except Exception:
                    pass
            except Exception:
                continue
        return {'status': 'ok', 'action': 'forget', 'id': memory_id,
                'note': 'deleted' if deleted else 'not found in ChromaDB'}
    except Exception as e:
        return {'status': 'error', 'action': 'forget', 'id': memory_id, 'error': str(e)}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MemPalace Enhanced CLI v4')
    subparsers = parser.add_subparsers(dest='cmd')
    
    p_search = subparsers.add_parser('search')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('--limit', type=int, default=5)
    p_search.add_argument('--no-mmr', dest='use_mmr', action='store_false', default=True)
    p_search.add_argument('--no-dedup', dest='dedup', action='store_false', default=True)
    p_search.add_argument('--no-strip', dest='strip', action='store_false', default=True)
    
    subparsers.add_parser('status')
    subparsers.add_parser('wake-up')
    
    p_mine = subparsers.add_parser('mine')
    p_mine.add_argument('--path', help='Path to mine')
    
    p_forget = subparsers.add_parser('forget')
    p_forget.add_argument('memory_id', help='Memory ID to forget')
    
    args = parser.parse_args()
    
    if args.cmd == 'search':
        result = cmd_search(args.query, args.limit, args.use_mmr, args.dedup, args.strip)
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
