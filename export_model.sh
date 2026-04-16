
# 捕获脚本执行结果
result=$(python export_int8_onnx.py)
exit_status=$?
if [[ $exit_status == 0 ]]; then
    cp mobileclip2_s2_text.onnx ../assets/models/
    cp mobileclip2_s2_visual.onnx ../assets/models/
    echo "✅ Model exported to ../assets/models/!"
else
    echo "❌ Model export failed!"
fi
