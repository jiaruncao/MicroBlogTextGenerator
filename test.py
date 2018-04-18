file = 'kobe.txt'
data = open(file,'r').read()
data = data.decode('utf-8')
chars = list(set(data)) #char vocabulary
data_size, _vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, _vocab_size)
char_to_idx = { ch : i for i,ch in enumerate(chars) }
#print (char_to_idx)
idx_to_char = { i : ch for i,ch in enumerate(chars) }
#print(idx_to_char)
context_of_idx = [char_to_idx[ch] for ch in data]

print context_of_idx

if __name__ == '__main__':
    print ('ok')