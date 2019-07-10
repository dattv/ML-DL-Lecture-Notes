# Prefix Sum

|__Algorithm 1: Prefix Sum__|
|---------------------------|
|Input: an array A          |
|<b>if</b> size of A is 1 <b>then</b>|
|<b>return</b> only element of A|
|<b>end</b>|
|Let A' is the sum of adjacent pair|
|Compute R'=AllPrefixSum(A')|
|Fill in missing entries of R' using another n/2 processor|


# Mergesort
|__Algorithm 2: Merge Sort__|
|---------------------------|
|<b>Input</b>: Array A with n elements|
|<b>Output</b>: Sorted A|
|n <- A|
|<b>If</b> n is 1 <b> then </b>|
|<b>return</b> only element of A|
|<b>end</b>|
|<b> else</b> |
|// in parallel|
|L <- Mergesort(A[0,...,n/2])|
|R <- Mergesort(A[n/2,...n])|
|<b>return</b> MERGE(L, R)|
|<b> end</b>|


# Merge
|__Algorithm 3: Merge__|
|----------------------|
|