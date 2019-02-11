class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print('Parent')

    def bar(self, message):
        print("%s from Parent" % message)

    def ts(self):
        self[self.parent].bar('hello')

class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类B的对象 FooChild 转换为类 FooParent 的对象
        super(FooChild, self).__init__()
        print('Child')

    def bar(self, message):
        super(FooChild, self).bar(message)
        print('Child bar fuction')
        print(self.parent)


if __name__ == '__main__':
    fooChild = FooChild()
    fooChild.bar('HelloWorld')
    # fooParent = FooParent()

# attn='use_flag'
# if attn>0:
#     print('1')



l = ['a','b','c','c','d','c']
find = 'c'
print([l[i] for i,v in enumerate(l) if v==find])

print( 2.74378770e-17   +2.15871614e-16 +  1.23689503e-12+
             8.17363885e-11  + 1.40273643e-11 +  9.66666448e-06+
             4.16512623e-11  + 8.35124414e-10  + 3.69707064e-04+
             9.99620557e-01)