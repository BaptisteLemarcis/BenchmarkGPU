#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cudnn.h>
#include "Trainer.h"

void main() {
	Trainer t(0,128);
	t.doStuff(0, 1);
	system("pause");
}