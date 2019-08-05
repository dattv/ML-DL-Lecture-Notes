import os

def rm_dir(input_folder):
    if os.path.isdir(input_folder) is True:
        list_ = os.listdir(input_folder)
        if len(list_) == 0:
            return os.rmdir(input_folder)

        else:
            list_folder = [os.path.join(input_folder, fo) for fo in list_ if
                           (os.path.isdir(os.path.join(input_folder, fo)) is True)]

            list_file = [os.path.join(input_folder, fi) for fi in list_ if
                         (os.path.isfile(os.path.join(input_folder, fi)) is True)]

            if len(list_folder) > 0:
                for folder in list_folder:
                    rm_dir(folder)

            if len(list_file) > 0:
                for file in list_file:
                    os.remove(file)

            rm_dir(input_folder)
