def lookAndSayQueries(q):
    # Write your code here
    result = []
    for i in range(len(q)):
        tmp = countnndSay(int(q[i]))
        print(tmp)
        results = list(map(int, tmp))
        result.append(sum(results))
    return result

def countnndSay(n): 
      
    # Base cases 
    if (n == 1): 
        return "1"
    if (n == 2): 
        return "11"
  
    s = "11" 
    for i in range(3, n + 1): 
          

        s += '$'
        l = len(s) 
  
        cnt = 1
        tmp = "" 

        for j in range(1 , l): 

            if (s[j] != s[j - 1]): 
                
                tmp += str(cnt + 0) 

                tmp += s[j - 1] 

                cnt = 1

            else: 
                cnt += 1

        s = tmp 
    return s; 
 
num = "1"
 

num = lookAndSayQueries(range(13))
print num
    