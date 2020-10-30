import os, shutil, h5py, numpy as np

master = r"C:\Users\mhasa\Desktop\CAD Repository"
master_dirs = os.listdir(master)

# delete all files without jpg or png
allowed_extensions = ['jpg', 'png']

# for category_folder in master_dirs:
#     source = os.path.join(master, f"{category_folder}", "CNN_Imgs",
#                           "12views_Icosa")
#     destination = os.path.join(master, f"{category_folder}")
#
#     try:
#         for folder_name in os.listdir(source):
#             if not folder_name.endswith('.txt'):
#                 file_list = os.listdir(os.path.join(source, folder_name))
#
#                 # now move the files
#                 for file in file_list:
#                     shutil.move(src=os.path.join(source, folder_name, file),
#                                 dst=destination)
#
#     except Exception as msg:
#         print(msg)


meta_data_master_folder = r"C:\Users\mhasa\Desktop\CAD_Repo_Metadat"
for category_folder in master_dirs:
    source = os.path.join(master, f"{category_folder}")
    try:
        for file in os.listdir(source):
            file_src_path = os.path.join(source, file)
            file_extension = file[-3:]

            if file_extension not in allowed_extensions:
                # print warning to user
                print(f"Invalid file found: {file_src_path}. Now moving it "
                      f"to metadata directory")

                # check if its a file
                # if os.path.isfile(file_src_path):
                #     os.remove(file_src_path)
                #     print("File Deleted")
                #
                # elif os.path.isdir(file_src_path):
                #     shutil.rmtree(file_src_path)
                #     print("Directory Deleted")
                # first create the metadata dir
                destination_to_move = os.path.join(meta_data_master_folder,
                                                   category_folder)

                if not os.path.exists(destination_to_move):
                    os.mkdir(destination_to_move)
                    print(
                        f"Destination Folder Created: {destination_to_move}.....")

                # now moving the file
                shutil.move(dst=destination_to_move,
                            src=file_src_path)

                print("=" * 50)

    except Exception as msg:
        print(msg)
#
# db = h5py.File(name=f"{master}" + r"//train_cad_10class.hdf5")
# print(db["labels"].shape)
# print(np.unique(db["labels"]))
#
