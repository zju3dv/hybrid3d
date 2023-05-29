def check_proj_labels():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
    import cv2
    print(input_rgb_np.shape, len(projection_pts_info))
    for idx, (coords, labels, H, W) in enumerate(projection_pts_info):
        rgb = (input_rgb_np[idx].transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        # print(rgb.shape)
        for ilabel in range(len(labels)):
            print(coords[ilabel])
            rgb = cv2.putText(rgb, '.{}.'.format(labels[ilabel]), (int(coords[ilabel][0]), int(coords[ilabel][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite('debug/{}-{}-{}.jpg'.format(fragment_pair_key, i, idx), rgb)
        # cv2.imwrite('debug/{}-{}-{}-heat.jpg'.format(fragment_pair_key, i, idx), (target_heatmap[idx] * 255).cpu().detach().numpy().transpose(1, 2, 0))
