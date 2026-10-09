#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "pwd.h"
#include "tempfile_port.h"
int main(int argc, char **argv) {
 if (argc > 1) {
  errno=0; assert(!getpwnam("alice"));
  assert(errno==(!strcmp(argv[1],"absent") ? 0 : !strcmp(argv[1],"long") ? ERANGE : EINVAL));
  puts("account rejection passed"); return 0;
 }
 struct passwd *p=getpwnam("alice"); assert(p && p->pw_uid==123 && !strcmp(p->pw_dir,"/home/alice"));
 p=getpwuid(123); assert(p && !strcmp(p->pw_name,"alice"));
 errno=7; assert(!getpwnam("absent") && errno==0);
 assert(!getlogin() && errno==ENXIO);
 char bad[]="/tmp/bad"; assert(mkstemp(bad)==-1 && errno==EINVAL && !strcmp(bad,"/tmp/bad"));
 char name[]="/tmp/proof-XXXXXX"; int fd=mkstemp(name); assert(fd>=0);
 struct stat st; assert(!fstat(fd,&st));
 assert(write(fd,"owned",5)==5); close(fd);
 FILE *file=tmpfile(); assert(file); assert(fputs("temporary",file)>=0); rewind(file); char text[16]={0};
 assert(fread(text,1,9,file)==9 && !strcmp(text,"temporary")); fclose(file);
 puts(name); puts("virtual account lookup, mkstemp, tmpfile: passed"); return 0;
}
