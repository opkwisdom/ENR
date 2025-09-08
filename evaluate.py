import torch, heapq
import numpy as np
def prepare_sorted_log_lists(probs):  # probs: (L=8, V=2048)
    logp = torch.log(probs.clamp_min(1e-12))  # (8, 2048)
    lists = []
    for t in range(logp.shape[0]):
        # 내림차순 정렬: [(logp, idx), ...]
        vals, idxs = torch.sort(logp[t], descending=True)
        lists.append([(float(vals[i].item()), int(idxs[i].item())) for i in range(len(idxs))])
    return lists

def topk_pairs(A, B, K):
    """
    A: [(sum_logp, seq_indices_tuple)] 혹은 [(logp, idx)] 중 왼쪽 컬렉션
    B: [(logp, idx)]  오른쪽 컬렉션
    반환: 상위 K쌍 (합, 결합된_정보)
    - 왼쪽이 '누적 시퀀스' 형태인 경우: seq + [idx]
    - 왼쪽이 '단일 토큰 리스트'이면: (idxA, idxB) 쌍
    구현은 일반화: 왼쪽 요소는 (scoreA, payloadA), 오른쪽은 (scoreB, payloadB)
    """
    # 내부 표현으로 통일
    def norm_list(lst):
        # 요소가 (score, payload) 형태가 아니면 payload=idx로 가정
        if isinstance(lst[0][1], (list, tuple)):
            return lst
        return [(lst[i][0], [lst[i][1]]) for i in range(len(lst))]

    A = norm_list(A)
    B = norm_list(B)

    # max-heap 흉내: 파이썬은 min-heap이므로 -score 사용
    heap = []
    visited = set()
    def push(i, j):
        if i < len(A) and j < len(B) and (i, j) not in visited:
            visited.add((i, j))
            score = A[i][0] + B[j][0]
            heapq.heappush(heap, (-score, i, j))

    push(0, 0)
    out = []
    while heap and len(out) < K:
        neg_s, i, j = heapq.heappop(heap)
        s = -neg_s
        seq = A[i][1] + B[j][1]  # payload 결합
        out.append((s, seq))
        # 이웃 확장 (정렬 가정: i+1, j / i, j+1 만 보면 됨)
        push(i+1, j)
        push(i, j+1)
    return out  # [(sum_logp, payload_list)]

def topk_ids_from_probs(probs, topk=100):
    """
    probs: (8, 2048) — 각 위치 확률
    반환: [(logp_sum, [idx0, ..., idx7])] 길이 topk
    """
    lists = prepare_sorted_log_lists(probs)  # 각 원소: [(logp, idx), ...] (내림차순)

    # step 0: 첫 위치 리스트를 '누적 시퀀스'로 초기화
    seqs = [(lp, [idx]) for lp, idx in lists[0]]

    # 이후 위치를 하나씩 병합하며 상위 topk만 유지
    for t in range(1, len(lists)):
        seqs = topk_pairs(seqs[:topk], lists[t], topk)  # 항상 상위 topk 시퀀스만 유지

    # seqs: [(sum_logp, [i0..i7])]
    return seqs[:topk]

# 사용 예시
def filter_sequences(candidates, allow_set, topk=100):
    """
    candidates: [(score, [idx0..idxL-1])] 리스트
    allow_set : set of tuple -> 허용된 시퀀스 집합
    topk      : 최종 topk 크기
    """
    filtered = [(s, seq) for s, seq in candidates if tuple(seq) in allow_set]
    return filtered[:topk]

def load_allowed_sequences(np_file):
    """
    np_file: .npy 경로, shape = (N, L)
    """
    arr = np.load(np_file)  # (N, L)
    # row 단위로 tuple로 변환 -> 빠른 lookup 가능
    allow_set = set(map(tuple, arr.tolist()))
    return allow_set


if __name__ == "__main__":
    # 예시: probs에서 top100 뽑기
    L, V = 8, 2048
    probs = torch.rand(L, V)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    top100 = topk_ids_from_probs(probs, topk=100)

    # numpy 파일에서 허용된 시퀀스 불러오기
    allow_set = load_allowed_sequences("../data/bge_base_en_v1.5/smtid_ms_full.npy")

    # top100 중 허용된 것만 필터링
    filtered_top = filter_sequences(top100, allow_set, topk=50)

    for s, seq in filtered_top[:5]:
        print("SEQ:", seq, "LOGP:", s, "PROB:", torch.exp(torch.tensor(s)).item())
        
