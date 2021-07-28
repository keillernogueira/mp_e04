import os
import shutil

def Separar_imagens(classes_interesse,novas_classes,path_imgs,path_labels,target_imgs,target_labels,mode=0,ext='jpg',pre = ""):
    # classes_interesse é uma lista de classes cujas imagens serão mantidas
    # novas_classes é uma lista das novos índices das classes de interesse
    # path_imgs é o caminho ou lista de caminhos para o diretório com as imagens
    # path_labels é o caminho para o diretório com as labels
    # target_imgs é o caminho para o diretório de destino das imagens
    # target_labels é o caminho para o diretório de destino das labels
    # mode indica 0 se haverá Separação ou 1 se somente haverá Contagem
    # ext indica a extensão dos arquivos de imagem
    # pre é o prefixo a ser adicionado nas imagens
    
    dct_clas = dict()
    for i,j in zip(classes_interesse,novas_classes):
        dct_clas[i] =j
        
    if mode ==0:
        if not os.path.exists(target_imgs):
            os.mkdir(target_imgs)
        if not os.path.exists(target_labels):
            os.mkdir(target_labels)
    
    list_files = os.listdir(path_labels)
    count_objs = dict()
    count_obj_img = dict()
    count_imgs = 0
    obj_img = []

    for file in list_files:
        move = False
        new_lines = []
        
        with open(os.path.join(path_labels, file),"r") as f:
            obj_img = []
            txt = f.readline()
            while(not (not txt)):
                classe = int(txt.split()[0])
                if classe in classes_interesse:
                    
                    if mode == 0:
                        new_class = dct_clas[classe]
                        line = txt.split()
                        line[0] = str(new_class)
                        line = ' '.join((n for n in line))+'\n'
                        new_lines.append(line)
                        
                    if classe not in obj_img:
                        try:
                            count_obj_img[classe] = count_obj_img[classe]+1
                        except:
                            count_obj_img[classe] = 1
                        obj_img.append(classe)
                    
                    try:
                        count_objs[classe] = count_objs[classe]+1
                    except:
                        count_objs[classe] = 1

                    move = True
                txt = f.readline()
                
        if move:
            count_imgs+=1
            if mode == 0:
                file_img = file[:-3]+ext
                target_img = os.path.join(target_imgs, pre+file_img)
                target_label = os.path.join(target_labels, pre+file)
                with open(target_label,"w") as f:
                    f.writelines(new_lines)
                
                if type(path_imgs) == list:
                    for p in path_imgs:
                        try:
                            path_img = os.path.join(p, file_img)
                            shutil.move(path_img,target_img)
                            break
                        except:
                            try:
                                file_img = file+ext.upper()
                                path_img = os.path.join(p, file_img)
                                shutil.move(path_img,target_img)
                                break
                            except:
                                try:
                                    file_img = file+ext[0].upper()+ext[1:]
                                    path_img = os.path.join(p, file_img)
                                    shutil.move(path_img,target_img)
                                    break
                                except:
                                    print("Colud not find",file,tuple([ext,ext.upper(),ext[0].upper()+ext[1:]]),"in ",p)
                                    continue
                else:
                    try:
                        path_img = os.path.join(path_imgs, file_img)
                        shutil.move(path_img,target_img)
                    except:
                        try:
                            file_img = file+ext.upper()
                            path_img = os.path.join(path_imgs, file_img)
                            shutil.move(path_img,target_img)
                            break
                        except:
                            try:
                                file_img = file+ext[0].upper()+ext[1:]
                                path_img = os.path.join(path_imgs, file_img)
                                shutil.move(path_img,target_img)
                                break
                            except:
                                print("Colud not find",file,tuple([ext,ext.upper(),ext[0].upper()+ext[1:]]))
                                continue

    print(count_objs)
    print(count_obj_img)
    print(count_imgs)