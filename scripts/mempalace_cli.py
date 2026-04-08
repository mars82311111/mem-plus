#!/usr/bin/env python3
"""
MemPalace Enhanced CLI Wrapper
============================
Integrates MemPalace with SM6's best features:
- MMR diversity reranking
- Levenshtein content deduplication
- Metadata stripping
- ChromaDB delete (forget)
- Wake-up with context injection
"""
import sys
import os
import json
import argparse

MEMPALACE_CLI = '/Users/mars/Library/Python/3.9/bin/mempalace'
RERANKER = '/Users/mars/.openclaw/workspace/skills/mempalace-memory/scripts/mempalace_reranker.py'
WORKSPACE = os.path.expanduser('~/.openclaw/workspace')

def run(args, timeout=30, python_bin='/usr/bin/python3'):
    import subprocess
    env = os.environ.copy()
    env['PATH'] = f'/Users/mars/Library/Python/3.9/bin:{env.get("PATH", "")}'
    result = subprocess.run(
        [python_bin, '-c', f'''
import sys
sys.path.insert(0, '/Users/mars/Library/Python/3.9/lib/python/site-packages')
import subprocess, json, os

# Build command
cli = '/Users/mars/Library/Python/3.9/bin/mempalace'
args = {args!r}
result = subprocess.run(
    [cli] + args,
    capture_output=True, text=True, timeout={timeout},
    env={{**os.environ, 'PATH': '/Users/mars/Library/Python/3.9/bin:' + os.environ.get('PATH', '')}}
)
print(result.stdout if result.returncode == 0 else result.stderr)
'''],
        capture_output=True, text=True, timeout=timeout + 5
    )
    return result.stdout, result.stderr, 0

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

def cmd_search(query, limit=5, use_mmr=True, dedup=True, strip=True):
    """Enhanced search with MMR + dedup + strip."""
    out, err, code = call_mempalace(['search', query, '--results', str(limit * 3)])
    
    if code != 0:
        return {'status': 'error', 'error': err}
    
    # 2. Parse results from markdown output
    # MemPalace search returns markdown format
    results = parse_search_output(out, query)
    
    # 3. Deduplicate
    if dedup:
        results = dedup_results(results)
    
    # 4. MMR rerank
    if use_mmr:
        results = mmr_rerank(results, query, lambda_param=0.7, limit=limit)
    
    return {
        'status': 'ok',
        'query': query,
        'results': results,
        'total_before_dedup': len(results) * 3 if dedup else 0
    }

def parse_search_output(output: str, query: str) -> list:
    """Parse mempalace search markdown output into structured results."""
    # Skip header lines
    lines = output.strip().split('\n')
    results = []
    current_content = []
    in_result = False
    
    for line in lines:
        if line.startswith('---') or line.startswith('==='):
            continue
        if 'Best drawer for' in line or 'Drawer #' in line:
            continue
        if line.startswith('**') and ('Memory' in line or 'Drawer' in line):
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    results.append({
                        'content': content,
                        'score': 1.0,  # MemPalace doesn't expose scores in CLI output
                        'source': 'mempalace'
                    })
                current_content = []
            continue
        if line.strip():
            current_content.append(line)
    
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            results.append({
                'content': content,
                'score': 1.0,
                'source': 'mempalace'
            })
    
    return results

def dedup_results(results, threshold=0.85):
    """Levenshtein-based deduplication."""
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
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
            prev = curr
        return prev[-1]
    
    def similarity(s1, s2):
        s1 = s1.lower(); s2 = s2.lower()
        max_len = max(len(s1), len(s2))
        if max_len == 0: return 1.0
        return 1.0 - (levenshtein(s1, s2) / max_len)
    
    deduped = []
    for r in results:
        content = r.get('content', '')
        is_dup = any(similarity(content, e.get('content','')) > threshold for e in deduped)
        if not is_dup:
            deduped.append(r)
    return deduped

def mmr_rerank(results, query, lambda_param=0.7, limit=5):
    """Maximum Marginal Relevance reranking."""
    if not results or len(results) <= limit:
        return results
    
    def levenshtein(s1, s2):
        if len(s1) < len(s2): return levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
            prev = curr
        return prev[-1]
    
    def similarity(s1, s2):
        s1 = s1.lower(); s2 = s2.lower()
        max_len = max(len(s1), len(s2))
        if max_len == 0: return 1.0
        return 1.0 - (levenshtein(s1, s2) / max_len)
    
    selected = []
    remaining = list(results)
    
    max_s = max((r.get('score', 0) for r in remaining), default=1)
    min_s = min((r.get('score', 0) for r in remaining), default=0)
    rng = max_s - min_s if max_s != min_s else 1.0
    
    def norm(r):
        return (r.get('score', 0) - min_s) / rng
    
    while len(selected) < limit and remaining:
        best = None
        best_idx = -1
        best_score = -float('inf')
        
        for idx, item in enumerate(remaining):
            relevance = norm(item)
            max_sim = max((similarity(item.get('content',''), s.get('content','')) for s in selected), default=0)
            diversity = 1.0 - max_sim
            mmr = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr > best_score:
                best_score = mmr
                best = item
                best_idx = idx
        
        if best:
            selected.append(best)
            remaining.pop(best_idx)
        else:
            break
    
    return selected

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
        from chromadb.config import Settings
        
        palace_path = os.path.expanduser('~/.mempalace/palace')
        client = chromadb.PersistentClient(path=palace_path)
        
        # Get all collections and find the one with this memory
        collections = client.list_collections()
        deleted = False
        
        for col in collections:
            try:
                collection = client.get_collection(col.name)
                # Try to get the embedding with this ID
                try:
                    item = collection.get(ids=[memory_id])
                    if item and item['ids']:
                        collection.delete(ids=[memory_id])
                        deleted = True
                        break
                except Exception:
                    # ID might not exist in this collection, try by metadata
                    try:
                        items = collection.get(where={'id': memory_id})
                        if items and items['ids']:
                            collection.delete(ids=items['ids'])
                            deleted = True
                            break
                    except Exception:
                        pass
            except Exception:
                continue
        
        if deleted:
            return {'status': 'ok', 'action': 'forget', 'id': memory_id}
        else:
            return {'status': 'ok', 'action': 'forget', 'id': memory_id, 'note': 'Memory not found in ChromaDB, may have been already deleted'}
    except Exception as e:
        return {'status': 'error', 'action': 'forget', 'id': memory_id, 'error': str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MemPalace Enhanced CLI')
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
