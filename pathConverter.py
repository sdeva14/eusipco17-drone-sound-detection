import sys, os;

fileDir = os.path.join('data', 'TUT-sound-events-2016-development');
filename = 'meta.txt';

inputFilePath = os.path.join(fileDir, filename);

pathFile = open(inputFilePath, 'r')
pathData = pathFile.read()
pathList = pathData.splitlines()

outputPath = os.path.join(fileDir, 'meta_converted.txt');
outputFile = open(outputPath, 'w');

converted = pathData.replace('/', '\\');
outputFile.write(converted);

# for path in pathList:
# 	splittedPath = path.split();
# 	splittedFile = splittedPath[0].split('/');

# 	converted = splittedFile[0] + '\\' + splittedFile[1]; 

pathFile.close()
outputFile.close()