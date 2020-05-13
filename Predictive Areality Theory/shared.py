import os

dbf = os.path.join(os.getcwd(), "database_files/")
pj = os.path.join(os.getcwd(), "picklejar/")

def parse_lang_feat(cat_id):
	parts = cat_id.split('-')
	assert(len(parts) == 2)
	return parts[0], int(parts[1])