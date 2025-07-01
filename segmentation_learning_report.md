# YOLACT Edgeのインスタンスセグメンテーション学習レポート

## 1. セグメンテーション領域の学習を行う箇所

### 1.1 Protonet（プロトタイプマスク生成ネットワーク）
- **場所**: `yolact_edge/yolact.py:1111-1121`
- **機能**: 画像全体のプロトタイプマスク（基底マスク）を生成
- **構造**: CNNベースのネットワークで、デフォルトで32個のプロトタイプマスクを出力

```python
# yolact.py: 1111-1121行目
if self.proto_src is None: in_channels = 3
elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
else: in_channels = self.backbone.channels[self.proto_src]
in_channels += self.num_grids

# The include_last_relu=false here is because we might want to change it to another function
self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)
```

### 1.2 マスク係数予測ヘッド
- **場所**: `yolact_edge/yolact.py:163` (PredictionModule内)
- **機能**: 各インスタンスごとのマスク係数を予測
- **出力**: 各アンカーボックスに対してmask_dim（32）次元の係数

```python
# yolact.py: 163行目
self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)
```

### 1.3 マスク生成プロセス
- **場所**: `yolact_edge/utils/output_utils.py:79-92`
- **手法**: Linear Combination (LINCOMB)
- **計算**: `masks = proto_data @ masks.t()` （プロトタイプマスクと係数の行列積）

```python
# output_utils.py: 79-92行目
if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
    # At this points masks is only the coefficients
    proto_data = dets['proto']
    
    # Test flag, do not upvote
    if cfg.mask_proto_debug:
        np.save('scripts/proto.npy', proto_data.cpu().numpy())
    
    if visualize_lincomb:
        display_lincomb(proto_data, masks)

    masks = proto_data @ masks.t()
    masks = cfg.mask_proto_mask_activation(masks)
```

## 2. Loss関数の実装

### 2.1 主要なLoss関数ファイル
- **メイン実装**: `yolact_edge/layers/modules/multibox_loss.py`
- **クラス**: `MultiBoxLoss`

### 2.2 マスクLossの種類

#### 2.2.1 Direct Mask Loss (`mask_type.direct`)
- **Loss関数**: Binary Cross Entropy
- **実装**: 
```python
F.binary_cross_entropy(torch.clamp(masks_p, 0, 1), masks_t, reduction='sum')
```

#### 2.2.2 Lincomb Mask Loss (`mask_type.lincomb`) - **デフォルト**
- **Loss関数**: Binary Cross Entropy (sigmoid activation時)
- **実装箇所**: `multibox_loss.py:531-534`

```python
if cfg.mask_proto_mask_activation == activation_func.sigmoid:
    pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')
else:
    pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')
```

### 2.3 マスクLoss計算の詳細プロセス

#### 2.3.1 GTマスクの前処理 (`multibox_loss.py:453-455`)
- 元画像サイズからProtonetの出力サイズ（138x138）にダウンサンプリング
- インターポレーションモード: bilinear

```python
downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                mode=interpolation_mode, align_corners=False)
```

#### 2.3.2 マスク予測の生成 (`multibox_loss.py:517-518`)
- プロトタイプマスクと係数の線形結合
- Sigmoid活性化関数の適用

```python
pred_masks = proto_masks @ proto_coef.t()
pred_masks = cfg.mask_proto_mask_activation(pred_masks)
```

#### 2.3.3 Loss正規化オプション
- `mask_proto_normalize_mask_loss_by_sqrt_area`: マスク面積の平方根で正規化
- `mask_proto_reweight_mask_loss`: 前景/背景ピクセルの重み付け
- `mask_proto_normalize_emulate_roi_pooling`: ROI poolingエミュレーション

### 2.4 補助的なセグメンテーションLoss

**セマンティックセグメンテーションLoss**
- **実装**: `multibox_loss.py:613-625`
- **Loss関数**: Binary Cross Entropy with Logits
- **目的**: 全体的なセグメンテーション精度の向上

```python
F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')
```

## 3. 学習時のLoss統合

### 3.1 Loss計算の流れ (`train.py`)

#### 3.1.1 Forward Pass (`train.py:431`)
```python
net_outs = net(images, extras=extras)
```

#### 3.1.2 Loss計算 (`train.py:325`)
```python
losses = criterion(out, targets, masks, num_crowds)
```

#### 3.1.3 Loss統合 (`train.py:333`)
```python
loss = sum([losses[k] for k in losses])
```

#### 3.1.4 Backpropagation (`train.py:336`)
```python
loss.backward()
```

### 3.2 Loss種類と重み付け

| Loss種類 | 記号 | デフォルト重み | 説明 |
|---------|------|---------------|------|
| Mask Loss | M | 6.125 | インスタンスマスクのメインLoss |
| Box Loss | B | 1.5 | バウンディングボックス回帰Loss |
| Class Loss | C | 1.0 | 分類Loss |
| Semantic Seg Loss | S | 1.0 | 補助的セグメンテーションLoss |
| Prototype Loss | P | - | プロトタイプマスクの正規化Loss |
| Diversity Loss | D | 1.5 | マスク係数の多様性を促すLoss |

### 3.3 Loss正規化
- **インスタンスLoss（B, C, M, D）**: 正例数で正規化
- **画像レベルLoss（P, S）**: バッチサイズで正規化

```python
# multibox_loss.py:186-194
total_num_pos = num_pos.data.sum().float()
for k in losses:
    if k not in ('P', 'E', 'S'):
        losses[k] /= total_num_pos  # 正例数で正規化
    else:
        losses[k] /= batch_size     # バッチサイズで正規化
```

## 4. 重要な設定パラメータ

```python
# config.py: 784-788行目
'mask_type': mask_type.lincomb,
'mask_alpha': 6.125,
'mask_proto_src': 0,
'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
'mask_proto_normalize_emulate_roi_pooling': True,
```

## 5. YOLACT Edgeの特徴的な実装ポイント

### 5.1 LINCOMBアプローチ
- 全体画像レベルのプロトタイプマスクを生成
- 各インスタンスは係数の線形結合で表現
- メモリ効率的で高速な推論が可能

### 5.2 FPNとの統合
- 複数スケールの特徴を活用
- Protonetは特定のFPN層から入力を受け取る

### 5.3 リアルタイム性能の最適化
- TensorRT対応のモジュール実装
- 効率的な行列演算による高速化

## まとめ

YOLACT Edgeは、効率的なLINCOMBアプローチを使用してインスタンスセグメンテーションを実現しています。プロトタイプマスクと係数の線形結合により、高速な推論を可能にしながら、Binary Cross Entropyベースの学習で高精度なセグメンテーションマスクを生成します。このアーキテクチャにより、エッジデバイスでのリアルタイム処理を実現しています。