import onnx_graphsurgeon as gs
import onnx
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple conversion for mondepth2 models.')

    parser.add_argument('--file', type=str)

    return parser.parse_args()

def main():
    args = parse_args()

    graph = gs.import_onnx(onnx.load(args.file))
    graph.toposort()

    onnx_model = gs.export_onnx(graph)

    onnx.save(onnx_model, args.file)

if __name__ == "__main__":
    main()