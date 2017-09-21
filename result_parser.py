import os

k_fold = 3

#
dirPath = 'C:\Users\Sdeva\Desktop'
# fname = '161020_exp_note.txt'
fname = 'result_exp_temp.txt'
filePath = os.path.join(dirPath, fname)

#
lines = [line.rstrip('\n') for line in open(filePath)]

#
ind = 1
precision_list = []
recall_list = []
fscore_list = []

temp_prec = []
temp_reca = []
temp_fsco = []

while ind < len(lines):
	curLine = lines[ind]
	if(curLine == 'Testing ----------------'):
		temp_fsco.append(float(lines[ind+1]))
		temp_prec.append(float(lines[ind+2]))
		temp_reca.append(float(lines[ind+3]))
		ind = ind + 3
	# end if

	if(len(curLine)<1 and len(temp_prec) > 0):
		fscore_list.append(sum(temp_fsco) / k_fold)
		precision_list.append(sum(temp_prec) / k_fold)
		recall_list.append(sum(temp_reca) / k_fold)
		
		temp_prec = []
		temp_reca = []
		temp_fsco = []
	# end if

	ind = ind + 1
# end while

# print result
print fscore_list
print '!'
print precision_list
print '!'
print recall_list
