p_FA_w = [0.        , 0.01848795, 0.0574447 , 0.11819082, 0.21030043,       0.33707494, 0.51799274, 0.73192473, 1.        ]
p_FA_b =    [0.        , 0.00436047, 0.0377907 , 0.08284884, 0.16133721,       0.31540698, 0.4622093 , 0.70639535, 1.        ]
p_FA_all = [0.        , 0.01587302, 0.05380683, 0.11164918, 0.20123756, 0.3330643 , 0.50766747, 0.72719935, 1.        ];
p_CD_w = [0.   , 0.144, 0.384, 0.572, 0.712, 0.928, 0.96 , 0.98 , 1.   ]
p_CD_b = [0.        , 0.08854167, 0.23958333, 0.38541667, 0.53645833,       0.64583333, 0.77604167, 0.88020833, 1.        ]
p_CD_all = [0.        , 0.1199095 , 0.32126697, 0.49095023, 0.63574661, 0.80542986, 0.8800905 , 0.93665158, 1.        ]
threshold_list = [88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5];

% Plot the results
figure(1)
plot(p_FA_w, p_CD_w, 'rs', p_FA_b, p_CD_b, 'ko', p_FA_all, p_CD_all, 'g.', ...
    'LineWidth',2, 'MarkerSize',10)
set(gca, 'FontSize', 16)
set(gca,'xlim',[-0.05, 1.05])
set(gca,'ylim',[-0.05, 1.05])
grid('on')
xlabel('Probability of False Alarm')
ylabel('Probability of Correct Detection')
hold on;
for i=1:length(threshold_list)
    % Put the threshold_list(i) on each dot, connect the white/Black points for 
    % that correspond to the same threshold_list(i).
    text(p_FA_w(i)-0.01, p_CD_w(i), num2str(threshold_list(i),'%.1f'), ...
        'HorizontalAlignment', 'right', 'FontSize', 14);
    text(p_FA_b(i)+0.01, p_CD_b(i), num2str(threshold_list(i),'%.1f'), ...
        'HorizontalAlignment', 'left', 'FontSize', 14);
    plot([p_FA_b(i),p_FA_w(i)], [p_CD_b(i),p_CD_w(i)], 'b-', 'LineWidth',2);
end
legend('White', 'Black', 'All','Location','southeast','FontSize',16)
