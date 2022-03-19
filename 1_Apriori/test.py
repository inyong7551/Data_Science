import sys

sup = float(sys.argv[1])
input_path = sys.argv[2]
f = open(input_path, 'r')
lines = []
while True:
    line = f.readline()
    if not line:
        break
    lines.append(line)
f.close()

before = len(lines)
lines = set(lines)
after = len(lines)
cnt = 0

for line in lines:
    li = line.split('\t')
    if float(li[2]) < sup:
        cnt += 1

print("----------[Deduplication Check]----------")
print("Before : " + str(before))
print("After : " + str(after))
print("----------[Minimum Support Check]----------")
print("Insufficient rule : " + str(cnt))
print("Finished.")
