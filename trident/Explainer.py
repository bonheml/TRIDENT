import torch


class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_names=('attn_drop', 'out_proj', 'to_out', 'attention_c'), discard_ratio=0.9):
        """An adaptation of the class proposed by [1] in https://github.com/jacobgil/vit-explain/blob/15a81d355a5aa6128ea4e71bbd56c28888d0f33b/vit_grad_rollout.py#L38
        :param model: the model to explain
        :param attention_layer_names: the name of the attention layers to target, defaults to 'attn_drop'
        :param discard_ratio: how many of the lowest attention values should be dropped, defaults to 0.9.
        This parameter is ignored when using Chefer method.

        References
        ----------
        [1] https://github.com/jacobgil/vit-explain
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.handles = []
        self.register_hooks(attention_layer_names)
        self.attentions = []
        self.attention_gradients = []

    def register_hooks(self, attention_layer_names):
        """Register forward and backward hook for each layer with 'attention_layer_name' and store them.

        :param attention_layer_names: the names of the target attention layers
        """
        for name, module in self.model.named_modules():
            for attention_layer_name in attention_layer_names:
                if attention_layer_name in name:
                    print(f"Registering hooks for layer {name}")
                    self.handles.append(module.register_forward_hook(self.get_attention))
                    self.handles.append(module.register_full_backward_hook(self.get_attention_gradient))

    def remove_hooks(self):
        """ Remove any existing hook and empty the handles list.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset_attention(self):
        """Reset the attention and attention gradient lists.
        """
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        """Add attention values obtained from forward hooks to the attention list.

        :param module: the target module
        :param input: the input received by the module
        :param output: the module output, here we are interested in the attention values.
        """
        self.attentions.append(output)

    def get_attention_gradient(self, module, grad_input, grad_output):
        """Add attention gradients obtained from backward hooks to the attention_gradients list.

        :param module: the target module
        :param grad_input: the gradient received as input, here we are interested in the attention gradient.
        :param grad_output: the gradient produced as output.
        """
        self.attention_gradients.append(grad_input[0])

    def grad_rollout_gildenblat(self):
        """ This is a slightly modified version of https://jacobgil.github.io/deeplearning/vision-transformer-explainability
        :return: mask of attention to add on the original image
        """
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(self.attentions, self.attention_gradients):
                weights = grad * 100
                attention_heads_fused = (attention * weights).mean(axis=1)
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        # Look at the total attention between the class token, and the image patches
        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        # mask = mask / np.max(mask)
        return mask

    def grad_rollout_chefer(self):
        """Implementation of the gradient rollout proposed in [1] for unimodal ViTs.
        See https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb
        for the original implementation.

        :return: attention mask to add on the original image

        References
        ----------
        [1] Chefer, Hila, Shir Gur, and Lior Wolf. "Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
        """
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(self.attentions, self.attention_gradients):
                # Eq (5) of [1], only positive contribution are kept before averaging
                attention_heads_fused = (grad * attention).clamp(min=0).mean(axis=1)[0]
                # Eq (6) of [1], accumulates the matrix relevancy at each layer
                result += torch.matmul(attention_heads_fused, result)
        I = torch.eye(result.size(-1))
        # compute \hat{R}^{qq} = R^{qq} - I, the matrix created by self-attention aggregation for Eq (9)
        result -= I
        # Eq (9) of [1], normalises the results to account equally for the influence of the token on itself
        # and for the contextualisation.
        result = result / result.sum(dim=-1) + I
        mask = result[0, 1:]
        return mask

    def __call__(self, input_tensor, weights, method="Chefer", device="cuda"):
        """Call the gradient rollout method using the forward and backward hooks previously registered.

        :param input_tensor: the input given to the model
        :param weights: the contribution of each latent representation to the output of the downstream task (e.g., archetype analysis).
        :param method: the gradient rollout method to use. Can be Gildenblat ou Chefer, defaults to 'Chefer'
        :return: an attention mask to add on the original image
        """
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_tensor, device)
        assert weights.size() == output.size()
        loss = (output * weights).sum()
        loss.backward()

        if method == "Gildenblat":
            return self.grad_rollout_gildenblat()
        elif method == "Chefer":
            return self.grad_rollout_chefer()
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")