file = open("data/obj.data", "w")
file.write("classes= 3\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = weights")
file.close()