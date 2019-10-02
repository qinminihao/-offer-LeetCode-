1. 二维数组中查找
	题目：
	在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
	思路：
	1. 暴力法
	class Solution:
    # array 二维列表
		def Find(self, target, array):
			# write code here
			if len(array) <= 0:
				return False
			for row in range(len(array)):
				for col in range(len(array[0])):
					if array[row][col] == target:
						return True
			return False
	
	
	2. #优化方法:选取数组右上角的数字，如果该数字等于要查找的数字，则查找结束；如果该数字大于要查找的数字，则剔除这个数字所在的列；小于则剔除行
	class Solution:
    # array 二维列表
		def Find(self, target, array):
			# write code here
			row = len(array)
			column = len(array[0])
			rows = 0
			columns = column - 1
			while(rows<row and columns>=0):
				if(array[rows][columns]==target):
					return True
				elif(array[rows][columns]>target):
					columns -= 1
				else:
					rows += 1
			return False
			
2. 替换空格
	题目：请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
	1. replace函数
	class Solution:
    # s 源字符串
		def replaceSpace(self, s):
			# write code here
			return s.replace(" ","%20")
			
	2. 循环
	class Solution:
    # s 源字符串
		def replaceSpace(self, s):
			# write code here
			result = ""
			for i in range(len(s)):
				if(s[i] != " "):
					result += s[i]
				else:
					result += "%20"
			return result
	
3. 从尾到头打印链表	
	题目：输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
	1. 从头到尾遍历，然后逆序即可
	class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
		def printListFromTailToHead(self, listNode):
			# write code here
			if listNode == None:
				return []
			ArrayList = []
			p = listNode
			while p != None:
				ArrayList.append(p.val)
				p = p.next
			ArrayList = ArrayList[::-1]
			return ArrayList
			
	2. 用insert函数将后面的节点插入到前面
	class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
		def printListFromTailToHead(self, listNode):
			# write code here
			ret = []
			pTmp = listNode
			while pTmp:
				ret.insert(0,pTmp.val)
				pTmp = pTmp.next
			return ret
	
	
4. 重建二叉树
	题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
			例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
	
	1. 递归
	class Solution:
    # 返回构造的TreeNode根节点
		def reConstructBinaryTree(self, pre, tin):
			# write code here
			if(len(pre)==0):
				return None
			else:
				flag = TreeNode(pre[0])
				flag.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],tin[:tin.index(pre[0])])
				flag.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],tin[tin.index(pre[0])+1:])
			return flag
	
5. 用两个栈实现队列
	题目：用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型
	1. stack1用于push数据，stack2用于pop数据
	class Solution:
		def __init__(self):
			self.stack1 = []
			self.stack2 = []
		def push(self,node):
			self.stack1.append(node)
			
		def pop(self):
			if self.stack2:
				return self.stack2.pop()
			else:
				while self.stack1:
					self.stack2.append(self.stack1.pop())
				return self.stack2.pop()
	
	
6. 	旋转数组的最小数字
	题目：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
			输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
			例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
			NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
	1. 暴力法
	class Solution:
		def minNumberInRotateArray(self, rotateArray):
			# write code here
			pre = -7e20
			for num in rotateArray:
				if num < pre :
					return num
				pre = num
				  
			if len(rotateArray) == 0:
				return 0
			return rotateArray[0]
			
	2. 二分查找法，定义首尾两个指针，还要考虑首尾中三个元素相等的特殊情况
	class Solution:
		def minNumberInRotateArray(self, rotateArray):
			# write code here
			if(len(rotateArray)==0):
				return 0
			else:
				high = len(rotateArray) - 1
				low = 0
				while(high - low > 1):
					mid = (high + low)//2
					if(rotateArray[high]==rotateArray[mid] and rotateArray[high]==rotateArray[low]):
						for i in range(low+1, high+1):
							if rotateArray[i] < rotateArray[low]:
								return rotateArray[i]
					else:
						if(rotateArray[mid]>=rotateArray[low]):
							low = mid
						if(rotateArray[mid]<=rotateArray[high]):
							high = mid
				return rotateArray[high]
	
	
7. 斐波那契数列
	题目：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）
	class Solution:
		def Fibonacci(self, n):
			# write code here
			if(n<=0):
				return 0
			if(n==1):
				return 1
			fib_one = 0
			fib_two = 1
			fib_mid = 0
			for i in range(n-1):
				fib_mid = fib_one + fib_two
				fib_one = fib_two
				fib_two = fib_mid
			return fib_mid
	
8. 青蛙跳台阶
	题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）
	class Solution:
		def jumpFloor(self, number):
			# write code here
			if number<=0:
				return 0
			num1 = 0
			num2 = 1
			 
			for i in range(1,number+1):
				skip = num1+num2
				num1=num2
				num2=skip
			return skip
	
	
9. 变态跳台阶
	题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
	#根据上一题目，一次可以跳1级，2级。则f(n) = f(n-1)+f(n-2) 
	#本题目f(n) = f(n-1)+f(n-2)+…+f(n-n)，这样计算起来非常麻烦。 
	#注意到，f(n-1) = f(n-2)+f(n-3)+…+f(n-n) 
	#两式相减得： 
	#f(n)-f(n-1) = f(n-1) 
	#调整：f(n) = 2*f(n-1) 

	class Solution:
		def jumpFloorII(self, number):
			# write code here
			return 2**(number-1)	
	
	
10 矩阵覆盖
	题目：我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
	class Solution:
		def rectCover(self, number):
			# write code here
			if number <=2:
				return number
			a = 1
			b = 2
			c = 0
			for i in range(number-2):
				c = a+b
				a = b
				b = c
			return c	
	
	
11. 二进制中1的个数
	题目：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
	1. 移位
	class Solution:
		def NumberOf1(self, n):
			# write code here
			count = 0
			flag = 1
			num = 0
			while(num<32):
				if(n&flag):
					count += 1
				flag = flag << 1
				num += 1
			return count
	2. n&(n-1)  每一个干掉末尾的一个1
	class Solution:
		def NumberOf1(self, n):
			count = 0
			n = 0xFFFFFFFF & n
			while(n):
				count += 1
				n = (n-1) & n
			return count 	
	
	
12. 数值的整数次方
	class Solution:
		def Power(self, base, exponent):
			# write code here
			flag = 0
			if base == 0:
				return False
			if exponent == 0:
				return 1
			if exponent < 0:
				flag = 1
			result = 1
			for i in range(abs(exponent)):
				result *= base
			if flag == 1:
				result = 1 / result
			return result
	
	
13 调整数组顺序使奇数位于偶数前面
	题目：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
			所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
	1. 开辟空间
	2. #冒泡排序思想，从第一个元素开始，往后遍历，经过第一次遍历后数组最后一个元素一定是偶数
		#所以下一次就不用遍历到最后一个元素了

	class Solution:
		def reOrderArray(self, array):
			# write code here
			for i in range(len(array)-1,0,-1):
				for j in range(i):
					if array[j]%2==0 and array[j+1]%2==1:
						array[j],array[j+1] = array[j+1], array[j]
			return array
	

14. 链表中倒数第k个节点
	#使用两个指针，同时都指向头结点，首先第一个指针走k-1步，即第一个结点到达第k个结点；然后两个指针同时往后移动，
	#直到第一个指针到达链表末尾时停止，这时候第二个指针的指向是倒数第k个结点，最后返回第二个指针指向的节点

	class Solution:
		def FindKthToTail(self, head, k):
			# write code here
			if not head or k<1:
				return None

			pre = head
			post = head

			# 首先将p1移动k-1步，到达第k个节点
			for _ in range(k-1):
				pre = pre.next
				if pre == None:    # 判断k的值是否大于链表长度，如果大于的话，p1会指向None
					return None

			# 当p1走到链表最后一个节点的时候，p2位于倒数第k个位置
			while pre.next!=None:
				pre = pre.next
				post = post.next
			return post


15. 反转链表，输出表头
	# class ListNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.next = None
	
	class Solution:
    # 返回ListNode
		def ReverseList(self, pHead):
			# write code here
			if not pHead or not pHead.next:
				return pHead
			cur = pHead
			pre = None
			while(cur.next != None):
				next_code = cur.next
				cur.next =pre
				pre = cur
				cur = next_code
			cur.next = pre
			return cur


16. 合并两个排序的链表
	题目：输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
	1. 递归
	class Solution:
    # 返回合并后列表
		def Merge(self, pHead1, pHead2):
			# write code here
			if(pHead1==None):
				return pHead2
			elif(pHead2==None):
				return pHead1
			if(pHead1.val<pHead2.val):
				pMerge = pHead1
				pMerge.next = self.Merge(pHead1.next,pHead2)
			if(pHead1.val>=pHead2.val):
				pMerge = pHead2
				pMerge.next = self.Merge(pHead1,pHead2.next)
			return pMerge
	2. 非递归
	class Solution:
    # 返回合并后列表
		def Merge(self, pHead1, pHead2):
			# write code here
			merge = ListNode(90)
			p = merge
			while(pHead1 and pHead2):
				if(pHead1.val>=pHead2.val):
					merge.next = pHead2
					pHead2 = pHead2.next
				elif(pHead1.val<pHead2.val):
					merge.next = pHead1
					pHead1 = pHead1.next
				merge = merge.next
			if pHead1:
				merge.next = pHead1
			elif pHead2:
				merge.next = pHead2
			return p.next


17. 树的子结构
	题目：输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
	class Solution:
		def HasSubtree(self, pRoot1, pRoot2):
			res=False
			if not pRoot1 or not pRoot2:
				return res
			if pRoot1.val==pRoot2.val:  #第一个判断条件，用来判断值相同时是否有可能为子结构
				res=self.Subtree(pRoot1,pRoot2)  
			if not res:  #左孩子与root2是否有可能
				res=self.HasSubtree(pRoot1.left,pRoot2)
			if not res:  #右孩子与root2是否有可能
				res=self.HasSubtree(pRoot1.right,pRoot2)
			return res
		 #当节点值相同时，才能进入底层判断
		def Subtree(self,pRoot1,pRoot2):
			if not pRoot2:  #如果root2为空，则提前结束，表示是子结构
				return True
			if not pRoot1:  #如果root1提前结束，表示不是子结构
				return False
			if pRoot2.val!=pRoot1.val:
				return False
			return self.Subtree(pRoot1.left,pRoot2.left) and self.Subtree(pRoot1.right,pRoot2.right) #只有对应左右子树相同时，才返回真


	class Solution:
		def HasSubtree(self, pRoot1, pRoot2):
			if pRoot1 == None or pRoot2 == None:
				return False
			
			def hasEqual(pRoot1,pRoot2):
				if pRoot1 == None:
					return False
				if pRoot2 == None:
					return True
				if pRoot1.val == pRoot2.val:
					if pRoot2.left == None:
						leftEqual = True
					else:
						leftEqual = hasEqual(pRoot1.left,pRoot2.left)
					if pRoot2.right == None:
						rightEqual = True
					else:
						rightEqual = hasEqual(pRoot1.right,pRoot2.right)
					return leftEqual and rightEqual
				return False

			if pRoot1.val == pRoot2.val:
				ret = hasEqual(pRoot1,pRoot2)
				if ret:
					return True
			
			ret = self.HasSubtree(pRoot1.left,pRoot2)
			if ret:
				return True
			ret = self.HasSubtree(pRoot1.right,pRoot2)
			return ret



18. 二叉树的镜像
	1. 非递归	
	class Solution:
    # 返回镜像树的根节点
		def Mirror(self, root):
			# write code here
			stack = []
			if root is None:
				return root
			else:
				stack.append(root)
				while len(stack) != 0:
					p = stack.pop()
					p.left, p.right = p.right, p.left
					 
					if p.left:
						stack.append(p.left)
					if p.right:
						stack.append(p.right)
	
	
	2. 递归
	class Solution:
    # 返回镜像树的根节点
		def Mirror(self, root):
			if not root:
				return None
			root.left, root.right = root.right, root.left
			if root.left:
				self.Mirror(root.left)
			if root.right:
				self.Mirror(root.right)

	
	
19. 顺时针打印矩阵	
	class Solution:
    # matrix类型为二维列表，需要返回列表
		def printMatrix(self, matrix):
			# write code here
			row = len(matrix)
			col = len(matrix[0])
			 
			left = 0
			right = col-1
			top = 0
			bottom = row-1
			res = []
			 
			while left <= right and top <= bottom:
				for i in range(left, right+1):
					res.append(matrix[top][i])
				for j in range(top+1, bottom+1):
					res.append(matrix[j][right])
				if top != bottom:
					for i in range(right-1,left-1,-1):
						res.append(matrix[bottom][i])
				if left != right:
					for i in range(bottom-1,top+1-1,-1):
						res.append(matrix[i][left])
				left += 1
				top += 1
				right -= 1
				bottom -= 1
			return res
	
	
20. 包含min函数的栈
	class Solution:
		def __init__(self):
			self.stack = []
			self.minValue = []
			
		def push(self, node):
			# write code here
			self.stack.append(node)
			if self.minValue:
				if self.minValue[-1] > node:
					self.minValue.append(node)
				else:
					self.minValue.append(self.minValue[-1])
			else:
				self.minValue.append(node)

		def pop(self):
			# write code here
			if self.stack == []:
				return None
			self.minValue.pop()
			return self.stack.pop()
		def top(self):
			# write code here
			if stack == []:
				return None
			return self.stack[-1]
		
		def min(self):
			# write code here
			if self.minValue == []:
				return None
			return self.minValue[-1]	
	
	
21. 栈的压入、弹出序列	
	题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序
	思路：如果下一个弹出的数字是栈顶数字，就直接弹出；否则把压栈序列中还没入栈的数字压入辅助栈，直到
			把下一个需要弹出的数字压入栈顶为止；如果所有数字都压入栈后仍没有找到下一个需要弹出的数字，则 return False
	class Solution:
		def IsPopOrder(self, pushV, popV):
			# write code here
			if not pushV or len(pushV) != len(popV):
				return False
			stack = []
			for i in pushV:
				stack.append(i)
				while len(stack) and stack[-1] == popV[0]:
					stack.pop()
					popV.pop(0)
			if len(stack):
				return False
			return True
	
22. 层次遍历，从上往下打印二叉树
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		# 返回从上到下每个节点值列表，例：[1,2,3]
		def PrintFromTopToBottom(self, root):
			# write code here
			l=[]
			if not root:
				return []
			q=[root]
			while len(q):
				t=q.pop(0)
				l.append(t.val)
				if t.left:
					q.append(t.left)
				if t.right:
					q.append(t.right)
			return l	
	
	
23. 二叉搜索树的后续遍历序列
	题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同
	思路：左子树上所有结点的值均小于根节点的值，右子树大于，序列最后一个是根节点
	class Solution:
		def VerifySquenceOfBST(self, sequence):
			# write code here
			if(len(sequence)==0):
				return False
			root = sequence[-1]
			left = []
			right = []
			for i in range(len(sequence)-1):
				if(sequence[i]>root):
					left.extend(sequence[0:i])
					right.extend(sequence[i:len(sequence)-1])
					break

			for i in range(len(right)):
				if(right[i]<root):
					return False    
			leftIs = True
			rightIs = True
			if(len(left)>0):
				leftIs = self.VerifySquenceOfBST(left)
			if(len(right)>0):
				rightIs = self.VerifySquenceOfBST(right)
			return leftIs and rightIs
	
	
	
24. 二叉树中和为某一个值的路径
	题目：输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
			路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径
	思路：1. 深度优先
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		# 返回二维列表，内部每个列表表示找到的路径
		def FindPath(self, root, expectNumber):
			# write code here
			if root == None:   # 这里一定要保证如果root为空，递归的时候就说明到达叶节点了
				return []
			res = []
			if root.val == expectNumber and root.left == None and root.right == None:
			#到达叶节点并且是叶节点的值已经与当前的目标值相等了
				res.append([root.val])
			# 继续对左右子树和剩下的目标值递归
			left = self.FindPath(root.left,expectNumber - root.val)
			right = self.FindPath(root.right,expectNumber - root.val)
			#这里不用担心root.left为空，如果为空的话，下一步就会停止递归，并返回[],然后再回到上一级
			for i in left + right:   
			# 这一步循环append保证了路径如果存在的话根节点的值一定会在子节点的值前面，
			# 而且left+right如果为空，那么这一步不会返回任何东西。
			# 这也保证了，如果叶节点的值不等于当前的目标值的话
			# 即这一条路径上的和不等于总的目标值，就不会将包含叶节点的路径返回。
				res.append([root.val]+i)
			return res
	
	2. 广度优先
	
	
25. 复杂链表的复制
	题目：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，
			另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head
	思路：三步：在旧链表中创建新的链表，根据旧链表的兄弟节点，初始化新链表的兄弟节点，从旧链表中拆分得到新链表
	# class RandomListNode:
	#     def __init__(self, x):
	#         self.label = x
	#         self.next = None
	#         self.random = None
	
	class Solution:
		# 返回 RandomListNode
		def Clone(self, pHead):
			if not pHead:
				return None
			#创建新链表
			pCur=pHead
			while pCur:
				pCur_next=pCur.next
				newnode=RandomListNode(pCur.label)
				newnode.next=pCur_next
				pCur.next=newnode
				pCur=pCur_next
			#改变新链表的random指针
			pCur=pHead
			while pCur:
				pCur_random=pCur.random
				newnode=pCur.next
				if pCur_random:
					newnode.random=pCur_random.next
				pCur=newnode.next
			pCur=pHead
			result=pHead.next
			#分离两链表
			while pCur:
				newnode=pCur.next
				pCur_next=newnode.next
				pCur.next=pCur_next
				if pCur_next:
					newnode.next=pCur_next.next
				else:
					newnode.next=None
				pCur=pCur.next
			return result	
	
	
26. 二叉搜索树与双向链表
	题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
	class Solution:
		def __init__(self):
			self.listHead = None
			self.listTail = None
		def Convert(self, pRootOfTree):
			if pRootOfTree==None:
				return
			self.Convert(pRootOfTree.left)
			if self.listHead==None:
				self.listHead = pRootOfTree
				self.listTail = pRootOfTree
			else:
				self.listTail.right = pRootOfTree
				pRootOfTree.left = self.listTail
				self.listTail = pRootOfTree
			self.Convert(pRootOfTree.right)
			return self.listHead	
	
	
27. 字符串的排列
	题目：输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来
			的所有字符串abc,acb,bac,bca,cab和cba。
	class Solution:
		def Permutation(self, ss):
			# write code here
			if len(ss)==0:return []
			if len(ss)==1: return [ss]
			ret = []
			l = list(ss)
			l.sort()
			ss = "".join(l)
			previous = None
			for i in range(len(ss)):
				if previous==ss[i]:
					continue
				else:
					previous = ss[i]
				res = self.Permutation(ss[:i]+ss[i+1:])
				for k in res:
					ret.append(ss[i:i+1]+k)
			return ret  			
			
			
28. 数组中次数超过一半的数字
	思路：1. 时间复杂度O(n) 空间复杂度O(n)
	class Solution:
		def MoreThanHalfNum_Solution(self, numbers):
			dict = {}
			for num in numbers:
				dict[num] = 1 if num not in dict else dict[num]+1
				if dict[num] > len(numbers)/2:
					return num
			return 0	
			
	2. 	时间复杂度O(n) 空间复杂度O(1),遇到相同的就抵消，剩下的数字就可能是大于一半的
	class Solution:
		def MoreThanHalfNum_Solution(self, numbers):
			count = 1
			number = numbers[0]
			for i in numbers[1:]:
				if number == i:
					count += 1
				else:
					count -= 1
					if count == 0:
						number = i
						count += 1
			# 至此选出一个数，但不一定次数过半。最后再迭代一轮，为其记个数。
			sum = 0
			for j in numbers:
				if j == number:
					sum += 1
	 
			return number if sum > len(numbers) // 2 else 0			
			
			
			
29. 最小的K个数
	思路：用最大堆，如果是找最大的K个数就用最小堆。最大堆的特点：某个节点的父节点为：(index - 1)//2
			父节点的左子节点为:(index*2+1),父节点的左子节点为:(index*2+2)
	class Solution:
		def GetLeastNumbers_Solution(self, tinput, k):
			# write code here

			#创建或者插入最大堆
			def createMaxHeap(num):
				maxHeap.append(num)
				currentIndex = len(maxHeap) - 1
				while currentIndex != 0:
					parentIndex = (currentIndex - 1) >> 1
					if maxHeap[parentIndex] < maxHeap[currentIndex]:
						maxHeap[parentIndex],maxHeap[currentIndex] = maxHeap[currentIndex],maxHeap[parentIndex]
					else:
						break
			#调整最大堆，头结点发生改变
			def adjustMaxHeap(num):
				if num < maxHeap[0]:
					maxHeap[0] = num
				maxHeapLen = len(maxHeap)
				index = 0
				while index < maxHeapLen:
					leftIndex = index * 2 + 1
					rightIndex = index * 2 + 2
					largerIndex = 0
					if rightIndex < maxHeapLen:
						if maxHeap[rightIndex] < maxHeap[leftIndex]:
							largerIndex = leftIndex
						else:
							largerIndex = rightIndex
					elif leftIndex < maxHeapLen:
						largerIndex = leftIndex
					else:
						break
						
					if maxHeap[index] < maxHeap[largerIndex]:
						maxHeap[index],maxHeap[largerIndex] = maxHeap[largerIndex],maxHeap[index]
					index = largerIndex
					
			maxHeap = []
			inputLen = len(tinput)
			if inputLen < k or k <= 0:
				return []
			for i in range(inputLen):
				if i < k:
					createMaxHeap(tinput[i])
				else:
					adjustMaxHeap(tinput[i])
			maxHeap.sort()
			return maxHeap
	
	或者用min函数
	class Solution:
		def GetLeastNumbers_Solution(self, tinput, k):
			# write code here
			if k > len(tinput):
				return []
			result = []
			for i in range(k):
				result.append(min(tinput))
				tinput.remove(min(tinput))
			return result


30. 连续子数组的最大和
	题目：例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和
	思路1：动态规划
	class Solution:
		def FindGreatestSumOfSubArray(self, array):
			# write code here
			#动态规划
			l = len(array)
			if l == 0:
				return 0
			# 以索引 i 结尾的最大子数组的和
			end_i_max = array[0]
			# 最后返回的数
			res = array[0]
			for i in range(1, l):
				# 例：[-3,1]
				end_i_max = max(array[i], end_i_max + array[i])
				res = max(res, end_i_max)
			return res
	思路2：
	class Solution:
		def FindGreatestSumOfSubArray(self, array):
			# write code here
			# 考虑特殊情况，数组的长度为0
			# 算法的时间复杂度为o(n)
			if len(array)<=0:
				return []
			temp_sum = 0
			list_sum = []
			for i in array:
				temp_sum = temp_sum + i
				list_sum.append(temp_sum)
				if temp_sum > 0:
					continue
				else:
					temp_sum = 0
			return max(list_sum)
			
	思路3：
	class Solution:
		def FindGreatestSumOfSubArray(self, array):
			if len(array)<=0:
				return []
			maxNum = array[0]
			tmpNum = 0
			for i in array:
				if tmpNum + i < i:
					tmpNum = i
				else:
					tmpNum += i
				if maxNum < tmpNum:
					maxNum = tmpNum
			return maxNum


31. 计算从1到n整数中1出现的次数
	思路1：转为字符串
	class Solution:
		def NumberOf1Between1AndN_Solution(self, n):
			# write code here
			ans = 0
			for i in range(1,n+1):
				ans += str(i).count('1')
			return ans

	思路2：不断与10取模
	class Solution:
		def NumberOf1Between1AndN_Solution(self, n):
			# write code here
			count = 0
			for num in range(1,n+1):
				while num > 0:
					if num % 10 == 1:
						count += 1
					num = num / 10
			return count


32. 把数组中的数字拼接为最小的数字
	class Solution:
		def PrintMinNumber(self, numbers):
			# write code here
			numbers =list(map(str,numbers))
			num = sorted(numbers,cmp = lambda x1,x2:(int(x1+x2)-int(x2+x1)))
			return "".join(num)

33. 丑数
	题目：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，
			因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
	#我们不需要每次都计算前面所有丑数乘以2，3，5的结果，然后再比较大小。因为在已存在的丑数中，一定存在某个数T2，
#在它之前的所有数乘以2都小于已有丑数，而T2×2的结果一定大于M，同理，也存在这样的数T3，T5，我们只需要标记这三个数即可。
	class Solution:
		def GetUglyNumber_Solution(self, index):
			# write code here
			if index<=0:
				return 0
			ugly = [1]
			index2 = index3 = index5 = 0
			while index>1:
				num = min(ugly[index2]*2,ugly[index3]*3,ugly[index5]*5)
				ugly.append(num)
				if num%2==0:
					index2+=1
				if num%3 ==0:
					index3+=1
				if num%5 ==0:
					index5+=1
				index-=1
			return ugly[-1]


34. 第一个只出现一次的字符
	题目：在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,
			并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
	class Solution:
		def FirstNotRepeatingChar(self, s):
			# write code here
			dict = {}
			for ele in s:
				dict[ele] = 1 if ele not in dict else dict[ele] + 1
			for i in range(len(s)):
				if dict[s[i]] == 1:
					return i
			return -1

35. 数组中的逆序对
	思路：就只是在guibing函数中多加一行cnt += len(left) - i ，去掉这行就是归并函数
	class Solution:
		def InversePairs(self, data):
			# write code here
			global cnt
			cnt = 0
			p = self.guibing(data)
			return cnt%1000000007
		def guibing(self,data):
			global cnt
			if len(data) == 1:
				return data
			mid = len(data)//2
			left = self.guibing(data[:mid])
			right = self.guibing(data[mid:])
			i = 0
			j = 0
			res = []
			while i < len(left) and j < len(right):
				if left[i] < right[j]:
					res.append(left[i])
					i += 1
				else:
					res.append(right[j])
					cnt += len(left) - i  #计算逆序对的数量
					j += 1
			res += left[i:]
			res += right[j:]
			return res


36. 两个链表的第一个公共节点
	# class ListNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.next = None

	#http://www.freesion.com/article/103969294/
	#当两个链表长度不等，且没有公共节点时，两个指针同时后移，当 $ p1 $ 指向 None 后，
	#将其重新指向第二个链表的第一个节点 $ p1=pHead2 $ 。当 $ p2 $ 指向 None 后，
	#将其重新指向第一个链表的第一个节点 $ p2=pHead1 $ 。当 $ p1=p2=None $ 时退出。

	class Solution:
		def FindFirstCommonNode(self, pHead1, pHead2):
			if not pHead1 or not pHead2:  
				return None  
			p1, p2 = pHead1, pHead2  
			while p1 != p2:  
				p1 = pHead2 if not p1 else p1.next  
				p2 = pHead1 if not p2 else p2.next  
			return p1 


37. 数字在排序数组中出现的次数
	class Solution:
		def GetNumberOfK(self, data, k):
			# write code here
			if not data:
				return 0
			low = 0
			high = len(data) - 1
			cnt = 0
			end = len(data) - 1
			while low <= high:
				medium = (low + high) / 2
				if data[medium] == k:
					# print "medium: ", medium
					# 这个数字前面后面都可能存在我们要找的数
					while medium >= 0 and data[medium] == k:
						medium -= 1
					medium += 1
					# print "left: ", medium
					while medium <= end and data[medium] == k:
						medium += 1
						cnt += 1
					return cnt
				elif data[medium] < k:
					low = medium + 1
				else:
					high = medium - 1
			return cnt


38. 二叉树的深度
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		def TreeDepth(self, pRoot):
			# write code here
			if not pRoot:
				return 0
			return 1+max(self.TreeDepth(pRoot.left),self.TreeDepth(pRoot.right))


39. 判断二叉树是否是平衡二叉树
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		def IsBalanced_Solution(self,root):
			if not root:
				return True
			left = self.GetDepth(root.left)
			right = self.GetDepth(root.right)
			if(abs(left-right)>1):
				return False
			return self.IsBalanced_Solution(root.left) and self.IsBalanced_Solution(root.right)
		
		def GetDepth(self,proot):
			if proot == None:
				return 0
			left = self.GetDepth(proot.left)
			right = self.GetDepth(proot.right)
			return max(left,right) + 1


40. 数组中只出现一次的数字
	题目：一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字
	思路1：
	class Solution:
		# 返回[a,b] 其中ab是出现一次的两个数字
		def FindNumsAppearOnce(self, array):
			# write code here
			dict = {}
			list = []
			for i in array:
				if i in dict:
					dict[i] += 1
				else:
					dict[i] = 1
			for key, value in dict.items():
				if value == 1:
					list.append(key)
			return list


	思路2：
	class Solution:
		# 相同的数异或为0，a^b^c = a^c^b（交换律），找到这两个数不同的位，以这一位将数组划分为两类，一类包含第一个数字，另一类包含另一个数字
		def FindNumsAppearOnce(self, array):
			# write code here
			if len(array) < 2:
				return None
			twoNumXor = None
			for num in array:
				if(twoNumXor == None):
					twoNumXor = num
				else:
					twoNumXor = twoNumXor ^ num
			count = 0
			 
			while twoNumXor % 2 == 0:
				twoNumXor = twoNumXor >> 1
				count += 1
				 
			temp = 1 << count
			 
			firstNum = None
			secondNum = None
			 
			for num in array:
				if temp & num == 0:
					if firstNum == None:
						firstNum = num
					else:
						firstNum = firstNum ^ num
				else:
					if secondNum == None:
						secondNum = num
					else:
						secondNum = secondNum ^ num
			return firstNum,secondNum


41. 和为s的连续正数序列
	思路：双指针
	class Solution:
		def FindContinuousSequence(self, tsum):
			# write code here
			if tsum < 3:
				return []
			small = 1
			big = 2
			curSum = small + big
			output = []
			middle = (tsum + 1) >> 1
			while small < middle:
				if curSum == tsum:
					output.append(range(small, big+1))
					big += 1
					curSum += big
				elif curSum < tsum:
					big += 1
					curSum += big
				else:
					curSum -= small
					small += 1
			return output


42. 和为s的两个数字
#根据中学的知识，当两条边越接近，面积越大（乘积越大）。由于从两头向中间进行查找的，
#找到的第一个组合一定是边差距最大的，所以乘积最小。
	class Solution:
		def FindNumbersWithSum(self, array, tsum):
			if not array: return []
			left, right = 0, len(array) - 1
			while left < right:
				_sum = array[left] + array[right]
				if _sum > tsum:
					right -= 1
				elif _sum < tsum:
					left += 1
				else:
					return [array[left], array[right]]
			return []

43. 左旋转字符串
	思路一：用一个temp记录需要移位的数字，不断移到最后面就可以了
	class Solution:
		def LeftRotateString(self, s, n):
			if s == '':
				return s
			ls = list(s)
			temp = 0 
			for i in range(n):
				temp = ls[0]
				del ls[0]
				ls.append(temp)
			return "".join(ls)

	思路二：切片
	class Solution:
		def LeftRotateString(self, s, n):
			return s[n:] + s[:n]

44. 反转单词顺序列
	class Solution:
		def ReverseSentence(self, s):
			# write code here
			if not s:
				return s
			s = list(s)
			self.reverse(s, 0, len(s) - 1)
			i,end = 0,0
			while i < len(s):
				if s[i] != ' ':
					start = i
					end = i + 1
					while end<len(s) and s[end]!=' ':
						end += 1
					self.reverse(s,start,end-1)
					i = end
				else:
					i += 1
			return "".join(s)

		def reverse(self,s,start,end):
			while start < end:
				s[start],s[end] = s[end],s[start]
				start += 1
				end -= 1

45. 扑克牌顺子
	#这5个数，除了0之外，不能有重复数字
	#除了0之外的数之间的差值，要小于数组中0的数量
	class Solution:
		def IsContinuous(self, numbers):
			# write code here
			if len(numbers)<5:
				return False
			numbers.sort()
			zeronum = numbers.count(0)
			num = 0
			for i in range(zeronum,len(numbers)-1):
				if numbers[i] == numbers[i+1]:
					return False
				num += (numbers[i+1] - numbers[i] -1) 
			if num > zeronum:
				return False
			else:
				return True


46. 圆圈中最后剩下的数
	class Solution:
		def LastRemaining_Solution(self, n, m):
			# write code here
			nums = list(range(n))
			if len(nums)==0:
				return -1
			ind = 0
			while len(nums)>1:
				ind = (ind+m-1)%(len(nums))
				nums = nums[:ind]+nums[ind+1:]
			return nums[0]
			
	思路2：
	class Solution:
		def LastRemaining_Solution(self, n, m):
			# write code here
			if n == 0 and m == 0:
				return -1
			child = [i for i in range(n)]
			p = 0
			cnt = 0
			while len(child) > 1:
				if cnt == m-1:
					child.pop(p)
					cnt = 0
				p = (p+1) % len(child)
				cnt += 1
			return child.pop()


47. 求1+2+3...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句
	class Solution:
		def Sum_Solution(self, n):
			# write code here
			return n>0 and self.Sum_Solution(n-1)+n

48. 不用加减乘除做加法
	class Solution:
		def Add(self, num1, num2):
			# write code here
			while num2:
				temp = num1^num2
				num2 = (num1&num2)<<1
				num1 = ((1<<32)-1)&temp
			return num1 if num1>>31 ==0 else num1-2**32


49. 把字符串转为整数
	思路1：
	class Solution:
		def StrToInt(self, s):
			# write code here
			if len(s)==0:
				return 0
			flag = 1
			if s[0] == "+":
				flag = 1
				s = s[1:]
			elif s[0] =='-':
				flag = -1
				s = s[1:]
			s_n = {str(x):x for x in range(10)}
			num = 0
			for i in range(len(s)):
				if s[i] not in s_n:
					return 0
				num =10*num+s_n[s[i]]
			return num*flag

	思路2：
	class Solution:
		def StrToInt(self, s):
			# write code here
			if not s:
				return 0
			res =0
			numbers = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
			for i in range(len(s)):
				if i==0 and (s[i] == '+' or s[i] == '-') :
					continue
				if s[i]<'0' or s[i]>'9':
					return False
				res = res * 10 + numbers[s[i]]
				 
			if s[0] == '-':
				res = 0 - res
			return res


50. 数组中重复的数字
	题目：在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。
			也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，
			那么对应的输出是第一个重复的数字2。

	思路1：
	#这种方法，不需要额外的存储空间，时间复杂度为O(n)，空间复杂度为o（1）！
	#由于这个给定的数组长度为n，所有的数字都在0—n-1之间，因此如果没有重复元素，那么排序之后的数组中的元素的值和它对应的
	#下标应该是相等的。因此以当前的数字numbers[i]为下标，都不会越界。把number[i]移动到numbers[numbers[i]]位置上，
	#表示坐标i与该坐标上的数字numbers[i]是相等的。利用这个方法，将每个数字都移动到相对应的坐标位置，移动时如果发现对应位置
	#上的数已经等于了numbers[i]，就表示出现了重复的数字。输出True即可，否则遍历完整个数组，仍然没有发现重复元素，返回False。

	class Solution:
		def duplicate(self, numbers, duplication):
			# write code here
			for i in range(len(numbers)):    
			# 每遍历一个位置，都要把这个位置上的数换成和下标相等的元素，或者是找到重复元素为止。
				while numbers[i] != i:
					if numbers[numbers[i]] == numbers[i]:
						duplication[0] = numbers[i]
						return True
					else:
						numbers[numbers[i]], numbers[i] = numbers[i], numbers[numbers[i]]
			return False

	思路2：
	class Solution:
		def duplicate(self, numbers, duplication):
			# write code here
			memories = []
			for i in numbers:
				if i not in memories:
					memories.append(i)
				else:
					duplication[0] = i
					return True
			return False
			
	思路3：
	class Solution:
		# 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
		# 函数返回True/False
		#先将输入的数组进行排序，再从头到尾遍历排序后的数组，如果相邻的两个元素相等，则存在重复数字。
		def duplicate(self, numbers, duplication):
			# write code here
			numbers = sorted(numbers)
			for i in range(1,len(numbers)):
				if numbers[i] == numbers[i-1]:
					duplication[0] = numbers[i]
					return True
			return False


51. 构建乘积数组
	题目：给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
	class Solution:
		def multiply(self, A):
			# write code here
			if len(A)<=1:
				return A
			BB = [1]*(len(A))
			BE = [1]*(len(A))
			for i in range(len(A)-1):
				BB[i+1] = BB[i]*A[i]
			for i in range(len(A)-2,-1,-1):
				BE[i] = BE[i+1]*A[i+1]
			B = []
			for i in range(len(A)):
				B.append(BB[i]*BE[i])
			return B


52. 正则表达式匹配
	题目：请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
		在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

	class Solution:
		# s, pattern都是字符串
		def match(self, s, pattern):
			# 递归终止条件
			if s == pattern: return True
			if not pattern: return False
			# 判断第二个字符是否为 *
			if len(pattern)>1 and pattern[1] == '*':
				# 判断首字符是否相等，不要忘记 . 的情况
				if s and (pattern[0] == s[0] or pattern[0] == '.'):
					# 相等的情况下，三种匹配模式
					return self.match(s[1:],pattern[2:]) \
							or self.match(s,pattern[2:]) \
							or self.match(s[1:], pattern)
				else:
					# 不等时，只能将pattern后移，继续判断
					return self.match(s, pattern[2:])
			# 第二个字符不为 *，直接对首字符进行比较    
			elif s and (s[0] == pattern[0] or pattern[0] == '.'):
				return self.match(s[1:], pattern[1:])
		
			return False


53. 表示数值的字符串
	题目：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，
		字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
	思路1：正则表达式
	class Solution:
		# s字符串
		def isNumeric(self, s):
			# write code here
			import re
			pattern = r'[\+|\-]?[0-9]*(\.[0-9]*)?([\e|\E][\+|\-]?[0-9]+)?$'
			if re.match(pattern,s):
				return True
			return False

	思路2：
	class Solution:
		# s字符串
		def isNumeric(self, s):
			# write code here
			try :
				p = float(s)
				return True
			except:
				return False


54. 字符流中第一个不重复的字符
	题目：请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，
			第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
			如果当前字符流没有存在出现一次的字符，返回#字符

	class Solution:
		# 返回对应char
		def __init__(self):
			self.s = ''
			self.dict = {}
		def FirstAppearingOnce(self):
			# write code here
			for i in self.s:
				if self.dict[i] == 1:
					return i
			return '#'
		def Insert(self, char):
			# write code here
			self.s += char
			if char in self.dict:
				self.dict[char] += 1
			else:
				self.dict[char] = 1


55. 链表中环的入口节点
	# class ListNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.next = None
	class Solution:
		def EntryNodeOfLoop(self, pHead):
			# write code here
			if pHead == None:
				return None
			fastPointer = pHead
			slowPointer = pHead
			
			while fastPointer and fastPointer.next:
				fastPointer = fastPointer.next.next
				slowPointer = slowPointer.next
				if fastPointer == slowPointer:
					break
			if fastPointer == None or fastPointer.next == None:
				return None
			#如果slow走了L的长度，则fast走了2L的长度
			#设从头开始到入口点的长度是s
			#slow在环里面走的长度是d
			#那么L = s + d
			#设环内slow没走的长度是M，则fast走的长度是 (M + d)*n + d + s = 2*L = 2*(s + d)
			#则s=m+(n-1)*(m+d)
			fastPointer = pHead
			while fastPointer != slowPointer:
				fastPointer = fastPointer.next
				slowPointer = slowPointer.next
			return fastPointer


56. 删除链表中重复的节点
	题目：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，
			返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5



57. 二叉树的下一个节点
	题目：给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回
	
	# class TreeLinkNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	#         self.next = None
	class Solution:
		def GetNext(self, pNode):
			# write code here
			#寻找右子树，如果存在就一直找到右子树的最左边，就是下一个结点
			#没有右子树，就寻找他的父节点，一直找到它是父节点的左子树，打印父节点
			if pNode.right:
				tmpNode = pNode.right
				while tmpNode.left:
					tmpNode = tmpNode.left
				return tmpNode
			else:
				tmpNode = pNode
				while tmpNode.next:
					if tmpNode.next.left == tmpNode:
						return tmpNode.next
					tmpNode = tmpNode.next
				return  None


58. 对称的二叉树
	题目：请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		def isSymmetrical(self, pRoot):
			# write code here
			def isMirror(left,right):
				if left == None and right == None:
					return True
				elif left == None or right == None:
					return False
				
				if left.val != right.val:
					return False
				ret1 = isMirror(left.left,right.right)
				ret2 = isMirror(left.right,right.left)
				return ret1 and ret2
			if pRoot == None:
				return True
			return isMirror(pRoot.left,pRoot.right)


59. 之字形打印二叉树
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		def Print(self, pRoot):
			# write code here
			if pRoot == None:
				return []
		 
			stack1 = [pRoot]
			stack2 = []
			ret = []
			
			while stack1 or stack2:
				if stack1:
					tmpRet = []
					while stack1:
						tmpNode = stack1.pop()
						tmpRet.append(tmpNode.val)
						if tmpNode.left:
							stack2.append(tmpNode.left)
						if tmpNode.right:
							stack2.append(tmpNode.right)
					ret.append(tmpRet)
						
				if stack2:
					tmpRet = []
					while stack2:
						tmpNode = stack2.pop()
						tmpRet.append(tmpNode.val)
						if tmpNode.right:
							stack1.append(tmpNode.right)
						if tmpNode.left:
							stack1.append(tmpNode.left)
					ret.append(tmpRet)
			return ret

60. 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		# 返回二维列表[[1,2],[4,5]]
		def Print(self, pRoot):
			# write code here
			if pRoot == None:
				return []
			queue1 = [pRoot]
			queue2 = []
			ret = []
			
			while queue1 or queue2:
				if queue1:
					tmpRet = []
					while queue1:
						tmpNode = queue1[0]
						tmpRet.append(tmpNode.val)
						del queue1[0]
						if tmpNode.left:
							queue2.append(tmpNode.left)
						if tmpNode.right:
							queue2.append(tmpNode.right)
					ret.append(tmpRet)
							
				if queue2:
					tmpRet = []
					while queue2:
						tmpNode = queue2[0]
						tmpRet.append(tmpNode.val)
						del queue2[0]
						if tmpNode.left:
							queue1.append(tmpNode.left)
						if tmpNode.right:
							queue1.append(tmpNode.right)
					ret.append(tmpRet)
			return ret  


61. 实现两个函数，分别用来序列化和反序列化二叉树
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		def Serialize(self, root):
			if not root:
				return '#'
			return str(root.val) +',' + self.Serialize(root.left) +','+ self.Serialize(root.right)

		def Deserialize(self, s):
			list = s.split(',')
			return self.deserializeTree(list)

		def deserializeTree(self, list):
			if len(list)<=0:
				return None
			val = list.pop(0)
			root = None
			if val != '#':
				root = TreeNode(int(val))
				root.left = self.deserializeTree(list)
				root.right = self.deserializeTree(list)
			return root


62. 二叉搜索树的第k个节点
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None
	class Solution:
		# 返回对应节点TreeNode
		def KthNode(self, pRoot, k):
			# write code here
			retList = []
			#中序遍历
			def preOrder(pRoot):
				if pRoot == None:
					return None
				preOrder(pRoot.left)
				retList.append(pRoot)
				preOrder(pRoot.right)
			
			preOrder(pRoot)
			if len(retList) < k or k < 1:
				return None

			return retList[k-1]


63. 数据流中的中位数
	class Solution:
		def __init__(self):
			self.l = []
		def Insert(self, num):
			# write code here
			self.l.append(num)
			self.l.sort()
			 
		def GetMedian(self,l):
			# write code here
			n = len(self.l)
			if n%2==1:
				return self.l[n//2]
			else:
				return (self.l[n//2-1]+self.l[n//2])/2.0


64. 滑动窗口的最大值
	class Solution:
		def maxInWindows(self, num, size):
			# 如果数组 num 不存在，则返回 []
			if not num:
				return []
			# 如果滑动窗口的大小大于数组的大小，或者 size 小于 0，则返回 []
			if size > len(num) or size <1:
				return []

			# 如果滑动窗口的大小为 1 ，则直接返回原始数组
			if size == 1:
				return num

			# 存放最大值，次大值的数组，和存放输出结果数组的初始化
			temp = [0]
			res = []

			# 对于数组中每一个元素进行判断
			for i in range(len(num)):
				# 判断第 i 个元素是否可以加入 temp 中
				# 如果比当前最大的元素还要大，清空 temp 并把该元素放入数组
				# 首先判断当前最大的元素是否过期
				if i -temp[0] > size-1:
					temp.pop(0)
				# 将第 i 个元素与 temp 中的值比较，将小于 i 的值都弹出
				while (len(temp)>0 and num[i] >= num[temp [-1]]):
					temp.pop()
				# 如果现在 temp 的长度还没有达到最大规模，将元素 i 压入
				if len(temp)< size-1:
					temp.append(i)
				# 只有经过一个完整的窗口才保存当前的最大值
				if i >=size-1:
					res.append(num[temp [0]])
			return res


65. 矩阵中的路径


66. 机器人的运动范围


67. 快排
	

68. 堆排序
	def  max_heapify(heap, heapSize, root):
		left = 2*root + 1
		right = 2*root + 2
		#父节点i的左子节点在位置(2*i+1);
		#父节点i的右子节点在位置(2*i+2);
		max_node = root
		
		if left < heapSize and heap[left] > heap[max_node]:
			max_node = left
		if right < heapSize and heap[right] > heap[max_node]:
			max_node = right
		if max_node != root:
			heap[max_node], heap[root] = heap[root], heap[max_node]
			max_heapify(heap, heapSize, max_node)
	 
	def build_max_heap(heap):
		n = len(heap)
		#从第一个非叶子节点处开始调整，一直调整到第一个根节点
		for i in range((n-2)//2, -1, -1):
			max_heapify(heap, n, i)
		return heap
	 
	def heap_sort(heap):
		heap = build_max_heap(heap)
		n = len(heap)
		alist = []
		for i in range(n-1, -1, -1):
			heap[0], heap[-1] = heap[-1], heap[0]
			alist.append(heap.pop(-1))
			max_heapify(heap, len(heap), 0)
		return alist
        

69. 有效的括号
	#思路：搜索到’(’，就入栈；搜索到‘)’，检查当前栈顶元素是否与之对应，出栈。最后栈空则全部匹配，中间不匹配或最后栈不空，则False
	class Solution:
		def isValid(self, s: str) -> bool:
			stack = []
			dit = {'(':1,')':-1,'[':2,']':-2,'{':3,'}':-3}
			if len(s)%2==1:
				return False
			for i in s:
				if dit[i]>0:
					stack.append(i)
				else:
					if stack:
						temp = stack.pop()
						if -dit[temp]!=dit[i]:
							return False
			return not stack

70. leetcode79题 单词搜索
	def exist(board,word):
		for i in range(len(word)):
			for j in range(len(word[0])):
				if search(i,j,0,board,word):
					return True
		return False

	def search(i,j,d,board,word):
		if i > len(word) - 1 or j > len(word[0]) - 1 or i <0 or j < 0:
			return False
		if board[i][j] != word[d]:
			return False
		if d == len(word) - 1:
			return True
		cur = board[i][j]
		board[i][j] = 0
		found = bool(search(i+1,j,d+1) or search(i-1,j,d+1) or search(i,j+1,d+1) or search(i,j-1,d+1))
		board[i][j] = cur
		return found


71.leetcode02题 链表两数之和
	# class ListNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.next = None

	class Solution:
		def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
			l3h = ListNode(0)
			l3 = l3h
			temps = 0 
			while l1 or l2 or temps:
				if l1:
					temps += l1.val
					l1 = l1.next
				if l2:
					temps += l2.val
					l2 = l2.next
				l3h.next = ListNode(temps%10)
				l3h = l3h.next
				temps = temps // 10
			return l3.next


72.leetcode78题 子集
	class Solution:
		def subsets(self, nums):
			"""
			:type nums: List[int]
			:rtype: List[List[int]]
			"""
			if not nums:
				return [[]]
			result = self.subsets(nums[1:])
			return result + [[nums[0]] + s for s in result]

73.leetcode22题 括号生成
	class Solution:
		def generateParenthesis(self, n: int) -> List[str]:
			if n == 0:
				return []
			result = []
			self.helper(n,n,'',result)
			return result
		
		def helper(self,l,r,item,result):
			if r < l:
				return
			if l == 0 and r == 0:
				result.append(item)
			if l > 0:
				self.helper(l-1,r,item + '(',result)
			if r > 0:
				self.helper(l,r-1,item + ')',result)


74.leetcode39题 组合总数
	class Solution:
		def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
			candidates.sort()
			n = len(candidates)
			res = []
			def helper(i, tmp_sum, tmp):
				if tmp_sum > target or i == n:
					return 
				if tmp_sum == target:
					res.append(tmp)
					return 
				helper(i,  tmp_sum + candidates[i],tmp + [candidates[i]])
				helper(i+1, tmp_sum ,tmp)
			helper(0, 0, [])
			return res




	