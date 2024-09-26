#day3 실습문제 2
def solution(t, p):
    answer = 0
    char_length = len(p)
    for i in range(len(t)):
        temp= t[i:i+char_length]
        if int(temp)<=int(p):
            answer+=1
        if i==len(t)-char_length:
            break

    return answer


t="3141592"
p="271"
print(solution(t,p))