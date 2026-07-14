"""
ONNX 模型量化完整流程 — ModelScope 版
pip install modelscope onnxruntime onnx
"""
import os
import glob
import onnx
import onnxruntime as ort
from modelscope.hub.snapshot_download import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "topdktu/PP_DoclayoutV2_onnx"
LOCAL_DIR = "./PP_DoclayoutV2_onnx"
OUTPUT_DIR = "./quantized_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 下载整个模型仓库到本地目录
#    默认缓存在 ~/.cache/modelscope/hub，也可指定 local_dir
print(f"📥 下载模型仓库 {MODEL_ID} ...")
model_dir = snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)
print(f"   已下载到: {model_dir}")

# 2. 找到所有 .onnx 文件
onnx_files = glob.glob(os.path.join(model_dir, "*.onnx"))
if not onnx_files:
    print("❌ 未找到 .onnx 文件，请检查 MODEL_ID 是否正确")
    exit(1)
print(f"   找到 {len(onnx_files)} 个 ONNX 文件: {[os.path.basename(f) for f in onnx_files]}")

for local_path in onnx_files:
    model_file = os.path.basename(local_path)
    print(f"\n🔧 处理 {model_file} ...")

    # 3. 检查模型 opset（需 >= 10）
    model = onnx.load(local_path)
    opset = model.opset_import[0].version
    print(f"   Opset 版本: {opset}")
    if opset < 10:
        print(f"   ⚠️ Opset {opset} < 10，量化可能失败，建议先升级 opset")
        continue

    # 4. 查看输入/输出信息
    session = ort.InferenceSession(local_path)
    print("   输入:")
    for inp in session.get_inputs():
        print(f"     - {inp.name}: {inp.shape} ({inp.type})")
    print("   输出:")
    for out in session.get_outputs():
        print(f"     - {out.name}: {out.shape} ({out.type})")

    # 5. 动态量化
    output_path = os.path.join(OUTPUT_DIR, model_file.replace(".onnx", "_uint8.onnx"))
    print("   ⚙️ 量化中 ...")
    quantize_dynamic(
        model_input=local_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    # 6. 对比大小
    orig_size = os.path.getsize(local_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    ratio = (1 - quant_size / orig_size) * 100
    print(f"   ✅ 完成: {orig_size:.1f} MB → {quant_size:.1f} MB (减小 {ratio:.1f}%)")