# Running the pipeline on AWS EC2

The pipeline is **CPU + RAM bound** (TF-IDF, fuzzy string features, LightGBM), so it needs a CPU instance, not a GPU.

Region: **US East (N. Virginia) `us-east-1`**. Every link below opens in that region.
If the console ever shows a different region (top-right), switch it back to "N. Virginia".

## 0. One-time account safety (5 min)
1. **Console home:** https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1
2. **Check your $100 credit is applied:** https://console.aws.amazon.com/billing/home#/credits
3. **Budget alarm:** https://console.aws.amazon.com/billing/home#/budgets
   → Create budget → **Customize (advanced)** → Cost budget → Next → Period *Monthly*, amount **$80**
   → **Advanced options: untick "Credits" (and "Refunds")** → Next → Add alert: threshold 100% of *Actual* → your email → Create.
   Unticking Credits matters. Otherwise the credit hides your usage and the alert only fires once real money is being charged.
   Add a second alert at 50% ($40) as an early warning. Billing data can lag a few hours, so don't rely on the alert
   alone; stop the instance when you're done.
4. **vCPU quota** (Running On-Demand Standard instances, which covers the m7i/c7i/r7i types):
   https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-1216C47A?region=us-east-1
   "Applied quota value" must be **≥ 16**. If it's lower: "Request increase at account level" → enter **32** → Request.
   New accounts can wait a few hours for approval, so do this first.
   Track the request: https://us-east-1.console.aws.amazon.com/servicequotas/home/requests?region=us-east-1
   (Later, only if you add a GPU cross-encoder: G-instance quota
   https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA?region=us-east-1 , needs ≥ 4 for g4dn.xlarge.)

## 1. Launch the instance (console)
Launch wizard: https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#LaunchInstances:

| Setting | Value |
|---|---|
| Name | `amazon-ml-er` |
| AMI | **Ubuntu Server 24.04 LTS** (x86_64) |
| Instance type | **m7i.4xlarge** (16 vCPU, 64 GB). Use r7i.4xlarge (128 GB) if you run out of memory; c7i.4xlarge (32 GB) is fine for a small dataset. |
| Key pair | Create new → `amazon-ml` → RSA → **.pem** → save it to `C:\Users\heman\.ssh\amazon-ml.pem` |
| Network | Allow SSH from **My IP** only |
| Storage | **100 GiB gp3** |

Launch, then copy the instance's **Public IPv4 address** from the instance list:
https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#Instances:

Cost in us-east-1 (on-demand Linux, approximate; the wizard shows the exact price):
m7i.4xlarge ≈ $0.81/hr (≈ 120 h for $100) · c7i.4xlarge ≈ $0.71/hr · r7i.4xlarge ≈ $1.06/hr.
From India, SSH to Virginia has about 250 ms lag, so typing feels slightly slow. Jobs running inside tmux are unaffected.
**Stop the instance whenever you're not running something.** A stopped instance only pays for disk (about $8/month for 100 GB).
Changing size later: Stop → Actions → Instance settings → Change instance type → Start.

## 2. Get the code and data onto the instance
From PowerShell in the repo folder on your laptop:
```powershell
.\aws\push.ps1 -Ip <PUBLIC_IP> -Key C:\Users\heman\.ssh\amazon-ml.pem
ssh -i C:\Users\heman\.ssh\amazon-ml.pem ubuntu@<PUBLIC_IP>
```
**Dataset: download it directly on the instance.** AWS bandwidth is far faster than uploading 700 MB+ from home.
In Chrome, open `chrome://downloads`, right-click the challenge zip, choose "Copy link address", then on the instance run:
```bash
wget -O ~/dataset.zip "<PASTED_LINK>"      # keep the quotes
```
(If the link has expired or needs a login, upload it instead: `.\aws\push.ps1 -Ip ... -Key ... -Dataset <path to zip>`.)

Then on the instance:
```bash
cd ~/er && bash aws/setup_ec2.sh ~/dataset.zip
```
This installs Python packages, unzips the data and finds the dataset folder automatically.

## 3. Run
```bash
tmux new -s er                 # keeps running if your SSH disconnects
bash aws/run_all.sh eda blocking   # 1) check the data + blocking recall first
bash aws/run_all.sh baseline       # 2) quick no-ML submission to confirm the format
bash aws/run_all.sh train predict  # 3) full model -> output/*.tsv (+ validator)
```
Detach with `Ctrl+b` then `d`. Reconnect later with `tmux attach -t er`. Watch CPU and RAM with `htop`.

## 4. Bring results back
On your laptop:
```powershell
.\aws\pull.ps1 -Ip <PUBLIC_IP> -Key C:\Users\heman\.ssh\amazon-ml.pem
```
Upload `output\matching_results.tsv` to the portal.

## 5. After editing code locally
Run `.\aws\push.ps1 ...` again. It re-uploads `src/` only; data and the Python environment on the instance stay.
Alternative: VS Code → install **Remote - SSH** → connect to `ubuntu@<IP>` with the .pem and edit on the instance directly.

## Notes
- The public IP **changes every time you stop and start** the instance. Copy the new one from the console.
- When the contest ends: EC2 → **Terminate** the instance, which also deletes its disk, so nothing keeps billing.
