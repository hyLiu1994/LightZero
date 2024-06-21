# coding=utf-8
import logging
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)


def consumer(left, right):
    print(hex(id(left)))
    print(hex(id(right)))
    left.close()  # 因为右管道要发送信息，还要接收信息，这里用不到左管道
    logging.info(f'>>> 右管道发送的信息：【你好】')
    right.send("【你好】")  # 1.给左管道发送信息
    while True:
        try:
            logging.info(f"\t\t右管道接收的信息：{right.recv()}")
        except EOFError:
            break


def main(ctx):
    left, right = ctx.Pipe()
    # print(hex(id(left)))
    # print(hex(id(right)))
    # consumer(left,right)
    ctx.Process(target=consumer, args=(left, right)).start()
    logging.info(f"\t\t左管道接收的信息：{left.recv()}")  # 2.接收右管道的信息

    right.close()  # 左管道要发信息，所以右管道关闭
    for i in range(10):
        logging.info(f'\t>>> 左管道发送的信息：【{i}个包子】')
        left.send(f'【{i}个包子】')
    left.close()  # 发信息完要关闭，因为后面不在发信息了

'''
资料参考: https://blog.csdn.net/mixintu/article/details/102164646
'''
if __name__ == '__main__':
    # windows 启动方式
    multiprocessing.set_start_method('spawn')
    # 获取上下文
    ctx = multiprocessing.get_context('spawn')
    # 检查这是否是冻结的可执行文件中的伪分支进程。
    ctx.freeze_support()
    main(ctx)

