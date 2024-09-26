def solution(n, lost, reserve):
    s = set(lost) & set(reserve)
    l = set(lost) - s #먼저 둘이 겹치는거 제거한 변수 설정
    r = set(reserve) - s
    print(s, l, r)
    for x in sorted(r):
        if x - 1 in l:
            l.remove(x - 1)
        elif x + 1 in l:
            l.remove(x + 1)

    return n - len(l)


print(solution(5, [2, 4], [1, 3, 5]))
