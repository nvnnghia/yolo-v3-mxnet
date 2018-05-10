import xml.etree.ElementTree as ET
import os

oname = ['person', 'car', 'bicycle', 'motorbike', 'dog']
def parse_rec(filename):
	tree = ET.parse(filename)
	root = tree.getroot()
	rmId = []
	for i, obj in enumerate(tree.findall('object')):
		if obj.find('name').text not in oname:
			rmId.append(i)
	a = len(rmId)-1
	while(a>=0):
		if (filename.split('/')[-1]).split('_')[0] in ['2007', '2008']:
			root.remove(root[rmId[a]+5])
		else:
			root.remove(root[rmId[a]+2])
		a =a -1

	objs = tree.findall('object')
	if objs != []:
		tree.write('VOC2012/Annotations/'+filename.split('/')[-1])
		file1.write((filename.split('/')[-1]).split('.')[0]+'\n')


with open('VOC2012/ImageSets/Main/trainval.txt', 'r') as f:
	lines = f.readlines()
imagenames = [x.strip() for x in lines]

file1 = open('trainvalremain.txt', 'w')
for imagename in imagenames:
	parse_rec('VOC2012/Annotations1/'+imagename + '.xml')

file1.close()

