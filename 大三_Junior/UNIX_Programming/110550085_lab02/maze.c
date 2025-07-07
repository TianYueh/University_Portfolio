/*
 * Lab problem set for UNIX programming course
 * by Chun-Ying Huang <chuang@cs.nctu.edu.tw>
 * License: GPLv2
 */
#include <linux/module.h>	// included for all kernel modules
#include <linux/kernel.h>	// included for KERN_INFO
#include <linux/init.h>		// included for __init and __exit macros
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/errno.h>
#include <linux/sched.h>	// task_struct requried for current_uid()
#include <linux/cred.h>		// for current_uid();
#include <linux/slab.h>		// for kmalloc/kfree
#include <linux/uaccess.h>	// copy_to_user
#include <linux/string.h>
#include <linux/device.h>
#include <linux/cdev.h>

#include "maze.h"

static dev_t devnum;
static struct cdev c_dev;
static struct class *clazz;

DEFINE_MUTEX(maze_mutex);

maze_t maze_of_users[3];
int user[3] = {-1, -1, -1};

//int arr[_MAZE_MAXX*_MAZE_MAXY];


static int maze_dev_open(struct inode *i, struct file *f) {
	printk(KERN_INFO "maze: device opened.\n");
	return 0;
}

static int maze_dev_close(struct inode *i, struct file *f) {
	

    int pid = current->pid;
    int num = -1;
    for(int i=0;i<_MAZE_MAXUSER;i++){
        if(user[i] == pid){
            num = i;
        }
    }
    if(num != -1){
        memset(maze_of_users[num].blk, 0, sizeof(maze_of_users[num].blk));
        maze_of_users[num].w = 0;
        maze_of_users[num].h = 0;
        maze_of_users[num].sx = 0;
        maze_of_users[num].sy = 0;
        maze_of_users[num].ex = 0;
        maze_of_users[num].ey = 0;
        maze_of_users[num].curx = 0;
        maze_of_users[num].cury = 0;
        user[num] = -1;
    }

    printk(KERN_INFO "maze: device closed.\n");
	return 0;
}

static ssize_t maze_dev_read(struct file *f, char __user *buf, size_t len, loff_t *off) {
	//printk(KERN_INFO "maze: read %zu bytes @ %llu.\n", len, *off);
    int p_id = current->pid;
    int num = -1;
    for(int i=0;i<_MAZE_MAXUSER;i++){
        if(user[i] == p_id){
            num = i;
            break;
        }
    }
    if(num == -1){
        return -EBADFD;
    }
    
    //int* arr = kmalloc(maze_of_users[num].w * maze_of_users[num].h * sizeof(int), GFP_KERNEL);
    char* arr = kmalloc(maze_of_users[num].w * maze_of_users[num].h, GFP_KERNEL);
    if(!arr){
        return -ENOMEM;
    }

    for(int i = 0;i<maze_of_users[num].h;i++){
        /*
        for(int j = 0;j<maze_of_users[num].w;j++){
            if(maze_of_users[num].blk[i][j]=='1'){
                arr[i*maze_of_users[num].w+j] = 1;
            }
            else{
                arr[i*maze_of_users[num].w+j] = 0;
            }
        }
        */
        memcpy(arr + i*maze_of_users[num].w, maze_of_users[num].blk[i], maze_of_users[num].w);
    }
    ssize_t ret = maze_of_users[num].h * maze_of_users[num].w;
    if(copy_to_user(buf+ *off, arr, maze_of_users[num].h * maze_of_users[num].w) != 0){
        kfree(arr);
        return -EBUSY;
    }
    *off += ret;
    kfree(arr);

	return maze_of_users[num].h * maze_of_users[num].w; 
}

static ssize_t maze_dev_write(struct file *f, const char __user *buf, size_t len, loff_t *off) {
	//printk(KERN_INFO "maze: write %zu bytes @ %llu.\n", len, *off);

    int p_id = current->pid;
    int num = -1;
    for(int i=0;i<_MAZE_MAXUSER;i++){
        if(user[i] == p_id){
            num = i;
            break;
        }
    }
    if(num == -1){
        return -EBADFD;
    }

    if(len % sizeof(coord_t) != 0){
        return -EINVAL;
    }

    
    
    coord_t *c_arr = kmalloc(len, GFP_KERNEL);
    if(copy_from_user(c_arr, (coord_t*)buf, len) != 0){
        kfree(c_arr);
        return -EBUSY;
    }   

    mutex_lock(&maze_mutex);
    size_t num_coords = len / sizeof(coord_t);
    for(size_t i = 0;i<num_coords;i++){
        coord_t mo = c_arr[i];
        int newx = maze_of_users[num].curx + mo.x;
        int newy = maze_of_users[num].cury + mo.y;
        if(newx < 0 || newx >= maze_of_users[num].w || newy < 0 || newy >= maze_of_users[num].h || maze_of_users[num].blk[newy][newx] == 1){
            continue;
        }
        maze_of_users[num].curx = newx;
        maze_of_users[num].cury = newy;

    }
    mutex_unlock(&maze_mutex);
    kfree(c_arr);

	return 0;
}


//Main Part for completion
static long maze_dev_ioctl(struct file *fp, unsigned int cmd, unsigned long arg) {
	//printk(KERN_INFO "maze: ioctl cmd=%u arg=%lu.\n", cmd, arg);
    coord_t c;
    //MAZE_CREATE
    if(cmd == MAZE_CREATE){


        if(copy_from_user(&c, (coord_t *)arg, sizeof(coord_t)) != 0){
            return -EBUSY;
        }
        //Error Handling
        if(c.x > _MAZE_MAXX || c.y > _MAZE_MAXY || c.x < 1 || c.y < 1){
            return -EINVAL;
        }
        int p_id = current->pid;
        for(int i = 0;i<_MAZE_MAXUSER;i++){
            if(user[i]==p_id){
                return -EEXIST;
            }
        }
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i]==-1){
                break;
            }
            if(i == _MAZE_MAXUSER-1){
                return -ENOMEM;
            }
        }



        //Start to create maze
        int user_id = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i]==-1){
                user[i] = p_id;
                user_id = i;
                break;
            }
        }

        maze_of_users[user_id].w = c.x;
        maze_of_users[user_id].h = c.y;
        maze_of_users[user_id].sx = c.x/2;
        maze_of_users[user_id].sy = c.y/2;
        maze_of_users[user_id].ex = c.x/2+1;
        maze_of_users[user_id].ey = c.y/2+1;
        maze_of_users[user_id].curx = c.x/2;
        maze_of_users[user_id].cury = c.y/2;

        if(maze_of_users[user_id].w == 3){
            maze_of_users[user_id].ex -= 1;
        }
        if(maze_of_users[user_id].h == 3){
            maze_of_users[user_id].ey -= 1;
        }

        for(int i = 0 ; i < c.y ; i++){
            for(int j = 0 ; j < c.x ; j++){
                if(i == 0 || i == c.y-1 || j == 0 || j == c.x-1){
                    maze_of_users[user_id].blk[i][j] = 1;
                }
                else{
                    maze_of_users[user_id].blk[i][j] = 0;
                }
            }
        }
        unsigned int nb = get_random_u32()%(c.y-1)+1;
        maze_of_users[user_id].blk[nb][1] = 1;
        //memset(maze_of_users[user_id].blk, 0, sizeof(maze_of_users[user_id].blk));
        //printk(KERN_INFO "maze: ioctl cmd=%u arg=%lu.\n", cmd, arg);
        return maze_of_users[user_id].w * maze_of_users[user_id].h;
    }
    //MAZE_RESET
    else if(cmd == MAZE_RESET){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        maze_of_users[num].curx = maze_of_users[num].sx;
        maze_of_users[num].cury = maze_of_users[num].sy;
        return 0;
    }
    //MAZE_DESTROY
    else if(cmd == MAZE_DESTROY){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        maze_of_users[num].w = 0;
        maze_of_users[num].h = 0;
        maze_of_users[num].sx = 0;
        maze_of_users[num].sy = 0;
        maze_of_users[num].ex = 0;
        maze_of_users[num].ey = 0;
        maze_of_users[num].curx = 0;
        maze_of_users[num].cury = 0;
        memset(maze_of_users[num].blk, 0, sizeof(maze_of_users[num].blk));
        user[num] = -1;
        return 0;
    }
    //MAZE_GETSIZE
    else if(cmd == MAZE_GETSIZE){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        coord_t c;
        c.x = maze_of_users[num].w;
        c.y = maze_of_users[num].h;
        if(copy_to_user((coord_t *)arg, &c, sizeof(coord_t)) != 0){
            return -EBUSY;
        }
        return 0;
    }
    //MAZE_MOVE
    else if(cmd == MAZE_MOVE){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        coord_t c;
        if(copy_from_user(&c, (coord_t *)arg, sizeof(coord_t)) != 0){
            return -EBUSY;
        }
        /*
        if(c.x == 0 && c.y == -1){
            if(maze_of_users[num].blk[maze_of_users[num].curx][maze_of_users[num].cury-1] == 0){
                maze_of_users[num].cury--;
                coord_t new_c;
                new_c.x = maze_of_users[num].curx;
                new_c.y = maze_of_users[num].cury;
                if(copy_to_user((coord_t *)arg, &new_c, sizeof(coord_t)) != 0){
                    return -EBUSY;
                }
            }
            else{
                return 0;
            }
        }
        else if(c.x == 0 && c.y == 1){
            if(maze_of_users[num].blk[maze_of_users[num].curx][maze_of_users[num].cury+1] == 0){
                maze_of_users[num].cury++;
                coord_t new_c;
                new_c.x = maze_of_users[num].curx;
                new_c.y = maze_of_users[num].cury;
                if(copy_to_user((coord_t *)arg, &new_c, sizeof(coord_t)) != 0){
                    return -EBUSY;
                }
            }
            else{
                return 0;
            }
        }
        else if(c.x == -1 && c.y == 0){
            if(maze_of_users[num].blk[maze_of_users[num].curx-1][maze_of_users[num].cury] == 0){
                maze_of_users[num].curx--;
                coord_t new_c;
                new_c.x = maze_of_users[num].curx;
                new_c.y = maze_of_users[num].cury;
                if(copy_to_user((coord_t *)arg, &new_c, sizeof(coord_t)) != 0){
                    return -EBUSY;
                }
            }
            else{
                return 0;
            }
        }
        else if(c.x == 1 && c.y == 0){
            if(maze_of_users[num].blk[maze_of_users[num].curx+1][maze_of_users[num].cury] == 0){
                maze_of_users[num].curx++;
                coord_t new_c;
                new_c.x = maze_of_users[num].curx;
                new_c.y = maze_of_users[num].cury;
                if(copy_to_user((coord_t *)arg, &new_c, sizeof(coord_t)) != 0){
                    return -EBUSY;
                }
            }
            else{
                return 0;
            }
        }
        else{
            return -EINVAL;
        }
        */
        int newx = maze_of_users[num].curx + c.x;
        int newy = maze_of_users[num].cury + c.y;
        if(newx < 0 || newx >= maze_of_users[num].w || newy < 0 || newy >= maze_of_users[num].h || maze_of_users[num].blk[newy][newx] == 1){
            return 0;
        }
        maze_of_users[num].curx = newx;
        maze_of_users[num].cury = newy;
        coord_t new_c;
        new_c.x = maze_of_users[num].curx;
        new_c.y = maze_of_users[num].cury;
        if(copy_to_user((coord_t *)arg, &new_c, sizeof(coord_t)) != 0){
            return -EBUSY;
        }

        return 0;
    }
    //MAZE_GETPOS
    else if(cmd == MAZE_GETPOS){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        coord_t c;
        c.x = maze_of_users[num].curx;
        c.y = maze_of_users[num].cury;
        if(copy_to_user((coord_t *)arg, &c, sizeof(coord_t)) != 0){
            return -EBUSY;
        }

        // seq_printf(m, " - Size [%d x %d]: (%d, %d) -> (%d, %d) @ (%d, %d)\n", maze_of_users[num].w, maze_of_users[num].h, maze_of_users[num].sx, maze_of_users[num].sy, maze_of_users[num].ex, maze_of_users[num].ey, maze_of_users[num].curx, maze_of_users[num].cury);



        return 0;
    }
    //MAZE_GETSTART
    else if(cmd == MAZE_GETSTART){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        coord_t c;
        c.x = maze_of_users[num].sx;
        c.y = maze_of_users[num].sy;
        if(copy_to_user((coord_t *)arg, &c, sizeof(coord_t)) != 0){
            return -EBUSY;
        }
        return 0;
    }
    //MAZE_GETEND
    else if(cmd == MAZE_GETEND){
        int p_id = current->pid;
        int num = -1;
        for(int i=0;i<_MAZE_MAXUSER;i++){
            if(user[i] == p_id){
                num = i;
                break;
            }
            if(i==_MAZE_MAXUSER-1){
                return -ENOENT;
            }
        }
        coord_t c;
        c.x = maze_of_users[num].ex;
        c.y = maze_of_users[num].ey;
        if(copy_to_user((coord_t *)arg, &c, sizeof(coord_t)) != 0){
            return -EBUSY;
        }
        return 0;
    }
    else{
        return -EINVAL;
    }



	return 0;
}

static const struct file_operations maze_dev_fops = {
	.owner = THIS_MODULE,
	.open = maze_dev_open,
	.read = maze_dev_read,
	.write = maze_dev_write,
	.unlocked_ioctl = maze_dev_ioctl,
	.release = maze_dev_close
};

static int maze_proc_read(struct seq_file *m, void *v) {
	//char buf[] = "`hello, world!` in /proc.\n";
	//seq_printf(m, buf);

    for(int i=0;i<_MAZE_MAXUSER;i++){
        seq_printf(m, "#%02d: ", i);
        if(user[i] == -1){
            seq_printf(m, "vacancy\n");
            continue;
        }
        else if(maze_of_users[i].w == 0){
            seq_printf(m, "vacancy\n");
            continue;
        }
        seq_printf(m, "pid %d - [%d x %d]: (%d, %d) -> (%d, %d) @ (%d, %d)\n", user[i], maze_of_users[i].w, maze_of_users[i].h, maze_of_users[i].sx, maze_of_users[i].sy, maze_of_users[i].ex, maze_of_users[i].ey, maze_of_users[i].curx, maze_of_users[i].cury);
        for(int j = 0;j<maze_of_users[i].h;j++){
            for(int k = 0;k<maze_of_users[i].w;k++){
                if(j==maze_of_users[i].cury && k==maze_of_users[i].curx){
                    seq_printf(m, "*");
                    continue;
                }
                else if(j==maze_of_users[i].sy && k==maze_of_users[i].sx){
                    seq_printf(m, "S");
                    continue;
                }
                else if(j==maze_of_users[i].ey && k==maze_of_users[i].ex){
                    seq_printf(m, "E");
                    continue;
                }
                else if(maze_of_users[i].blk[j][k] == 1){
                    seq_printf(m, "#");
                }
                else{
                    seq_printf(m, ".");
                }
            }
            seq_printf(m, "\n");
        }


    }
	return 0;
}

static int maze_proc_open(struct inode *inode, struct file *file) {
	return single_open(file, maze_proc_read, NULL);
}

static const struct proc_ops maze_proc_fops = {
	.proc_open = maze_proc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};

static char *maze_devnode(const struct device *dev, umode_t *mode) {
	if(mode == NULL) return NULL;
	*mode = 0666;
	return NULL;
}

//Change all hello_mod to maze
static int __init maze_init(void)
{
	// create char dev
	if(alloc_chrdev_region(&devnum, 0, 1, "updev") < 0)
		return -1;
	if((clazz = class_create("upclass")) == NULL)
		goto release_region;
	clazz->devnode = maze_devnode;
	if(device_create(clazz, NULL, devnum, NULL, "maze") == NULL)
		goto release_class;
	cdev_init(&c_dev, &maze_dev_fops);
	if(cdev_add(&c_dev, devnum, 1) == -1)
		goto release_device;

	// create proc
	proc_create("maze", 0, NULL, &maze_proc_fops);

	printk(KERN_INFO "maze: initialized.\n");
	return 0;    // Non-zero return means that the module couldn't be loaded.

release_device:
	device_destroy(clazz, devnum);
release_class:
	class_destroy(clazz);
release_region:
	unregister_chrdev_region(devnum, 1);
	return -1;
}

static void __exit maze_cleanup(void)
{
	remove_proc_entry("maze", NULL);

	cdev_del(&c_dev);
	device_destroy(clazz, devnum);
	class_destroy(clazz);
	unregister_chrdev_region(devnum, 1);

	printk(KERN_INFO "maze: cleaned up.\n");
}



module_init(maze_init);
module_exit(maze_cleanup);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Chun-Ying Huang");
MODULE_DESCRIPTION("The unix programming course demo kernel module.");
