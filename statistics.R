library(lme4)

########### Tests for confirming training accuracy #############

x = read.csv('human-model-comparison.csv')

x_TM = subset(x, comparison=='TM')
x_TS = subset(x, comparison=='TS')
x_MS = subset(x, comparison=='MS')

x_human_TMTS = subset(subset(x, who=='human'), comparison == 'TM' | comparison == 'TS')
x_human_TMMS = subset(subset(x, who=='human'), comparison == 'TM' | comparison == 'MS')
x_human_TSMS = subset(subset(x, who=='human'), comparison == 'TS' | comparison == 'MS')

x_model_TMTS = subset(subset(x, who=='model'), comparison == 'TM' | comparison == 'TS')
x_model_TMMS = subset(subset(x, who=='model'), comparison == 'TM' | comparison == 'MS')
x_model_TSMS = subset(subset(x, who=='model'), comparison == 'TS' | comparison == 'MS')

# whether there is a difference between the model and the human data differences
dist = lmer(value ~ who + (1|subject), data=x_TM, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_TM, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ who + (1|subject), data=x_TS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_TS, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ who + (1|subject), data=x_MS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_MS, REML=FALSE)
anova(dist, none)

# whether within the human data there is a difference between conditions
dist = lmer(value ~ comparison + (1|subject), data=x_human_TMTS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_human_TMTS, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ comparison + (1|subject), data=x_human_TMMS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_human_TMMS, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ comparison + (1|subject), data=x_human_TSMS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_human_TSMS, REML=FALSE)
anova(dist, none)

# whether within the model data there is a difference between conditions
dist = lmer(value ~ comparison + (1|subject), data=x_model_TMTS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_model_TMTS, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ comparison + (1|subject), data=x_model_TMMS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_model_TMMS, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ comparison + (1|subject), data=x_model_TSMS, REML=FALSE)
none = lmer(value ~ 1 + (1|subject), data=x_model_TSMS, REML=FALSE)
anova(dist, none)

########### Tests for Experiment 1 #############
# Test for which value of H_prior the subject-wise difference between the standard network and the network with altered H_prior fit the social/mechanical or the social/individual difference in human data

human_df = subset(x, select=-X)
levels(human_df$comparison) <- c("MS", "TM", "TS", "1", "0.5", "0.45", "0.4", "0.35", "0.3", "0.25", "0.2", "0.15", "0.1", "0.09", "0.08", "0.07", "0.06", "0.05")
# this is necessary because the distance has been computed in the inverse way in human data above
human_df$value = -1 * human_df$value

prior0 = read.csv('./results/RI_difference_model_hyper-prior_net-0.csv', sep="\t")
prior1 = read.csv('./results/RI_difference_model_hyper-prior_net-1.csv', sep="\t")
prior2 = read.csv('./results/RI_difference_model_hyper-prior_net-2.csv', sep="\t")
prior3 = read.csv('./results/RI_difference_model_hyper-prior_net-3.csv', sep="\t")
prior4 = read.csv('./results/RI_difference_model_hyper-prior_net-4.csv', sep="\t")
prior5 = read.csv('./results/RI_difference_model_hyper-prior_net-5.csv', sep="\t")
prior6 = read.csv('./results/RI_difference_model_hyper-prior_net-6.csv', sep="\t")
prior7 = read.csv('./results/RI_difference_model_hyper-prior_net-7.csv', sep="\t")
prior8 = read.csv('./results/RI_difference_model_hyper-prior_net-8.csv', sep="\t")
prior9 = read.csv('./results/RI_difference_model_hyper-prior_net-9.csv', sep="\t")

df <- rbind(human_df, prior0)
df <- rbind(df, prior1)
df <- rbind(df, prior2)
df <- rbind(df, prior3)
df <- rbind(df, prior4)
df <- rbind(df, prior5)
df <- rbind(df, prior6)
df <- rbind(df, prior7)
df <- rbind(df, prior8)
df <- rbind(df, prior9)

subset_MSa = subset(df, comparison == "MS" | comparison == "0.45")
subset_MSb = subset(df, comparison == "MS" | comparison == "0.4")
subset_MSc = subset(df, comparison == "MS" | comparison == "0.35")

subset_TSa = subset(df, comparison == "TS" | comparison == "0.15")
subset_TSb = subset(df, comparison == "TS" | comparison == "0.1")
subset_TSc = subset(df, comparison == "TS" | comparison == "0.09")

# whether the difference values differ significantly as a function of the identity (model or human)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSa, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSa, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSb, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSb, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSc, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSc, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSa, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSa, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSb, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSb, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSc, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSc, REML=FALSE)
anova(dist, none)


# Test for which value of H_sensor the subject-wise difference between the standard network and the network with altered H_sensor fit the social/mechanical or the social/individual difference in human data

human_df = subset(x, select=-X)
levels(human_df$comparison) <- c("MS", "TM", "TS", "0.001", "0.002", "0.0022", "0.0025", "0.0029", "0.0033", "0.004", "0.005", "0.0067", "0.01", "0.0111", "0.0125", "0.0143", "0.0167", "0.02")
# this is necessary because the distance has been computed in the inverse way in human data above
human_df$value = -1 * human_df$value

sensor0 = read.csv('./results/RI_difference_model_hyper-sensor_net-0.csv', sep="\t")
sensor1 = read.csv('./results/RI_difference_model_hyper-sensor_net-1.csv', sep="\t")
sensor2 = read.csv('./results/RI_difference_model_hyper-sensor_net-2.csv', sep="\t")
sensor3 = read.csv('./results/RI_difference_model_hyper-sensor_net-3.csv', sep="\t")
sensor4 = read.csv('./results/RI_difference_model_hyper-sensor_net-4.csv', sep="\t")
sensor5 = read.csv('./results/RI_difference_model_hyper-sensor_net-5.csv', sep="\t")
sensor6 = read.csv('./results/RI_difference_model_hyper-sensor_net-6.csv', sep="\t")
sensor7 = read.csv('./results/RI_difference_model_hyper-sensor_net-7.csv', sep="\t")
sensor8 = read.csv('./results/RI_difference_model_hyper-sensor_net-8.csv', sep="\t")
sensor9 = read.csv('./results/RI_difference_model_hyper-sensor_net-9.csv', sep="\t")

df <- rbind(human_df, sensor0)
df <- rbind(df, sensor1)
df <- rbind(df, sensor2)
df <- rbind(df, sensor3)
df <- rbind(df, sensor4)
df <- rbind(df, sensor5)
df <- rbind(df, sensor6)
df <- rbind(df, sensor7)
df <- rbind(df, sensor8)
df <- rbind(df, sensor9)

subset_MSa = subset(df, comparison == "MS" | comparison == "0.0022")
subset_MSb = subset(df, comparison == "MS" | comparison == "0.0025")
subset_MSc = subset(df, comparison == "MS" | comparison == "0.0029")

subset_TSa = subset(df, comparison == "TS" | comparison == "0.0067")
subset_TSb = subset(df, comparison == "TS" | comparison == "0.01")
subset_TSc = subset(df, comparison == "TS" | comparison == "0.0111")

# whether the difference values differ significantly as a function of the identity (model or human)

dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSa, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSa, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSb, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSb, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_MSc, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_MSc, REML=FALSE)
anova(dist, none)

dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSa, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSa, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSb, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSb, REML=FALSE)
anova(dist, none)
dist = lmer(value ~ who + (1|subject) + (1|net), data=subset_TSc, REML=FALSE)
none = lmer(value ~ 1 + (1|subject) + (1|net), data=subset_TSc, REML=FALSE)
anova(dist, none)


########### Tests for Experiment 2 #############

x = read.csv('results/network_summary/wLaP_statistics_norm.csv', sep='\t')

data_t0 = subset(x, timestep==0)
data_t21 = subset(x, timestep==21)

# between tablet and mechanical
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t0, condition!=2))
summary(dist.lm)

# between tablet and social
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t0, condition!=1))
summary(dist.lm)

# between mechanical and social
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t0, condition!=0))
summary(dist.lm)

# between tablet and mechanical
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t21, condition!=2))
summary(dist.lm)

# between tablet and social
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t21, condition!=1))
summary(dist.lm)

# between mechanical and social
dist.lm = lm(distance ~ condition + (1|net), data=subset(data_t21, condition!=0))
summary(dist.lm)


y = read.csv('results/network_summary/wLaP-between_statistics.csv', sep='\t')

data_t0 = subset(y, timestep==0)
data_t21 = subset(y, timestep==21)

# between T-M and T-S
dist.lm = lm(distance ~ condition, data=subset(data_t0, condition!=2))
summary(dist.lm)

# between T-M and M-S
dist.lm = lm(distance ~ condition, data=subset(data_t0, condition!=1))
summary(dist.lm)

# between T-S and M-S
dist.lm = lm(distance ~ condition, data=subset(data_t0, condition!=0))
summary(dist.lm)



# between T-M and T-S
dist.lm = lm(distance ~ condition, data=subset(data_t21, condition!=2))
summary(dist.lm)

# between T-M and M-S
dist.lm = lm(distance ~ condition, data=subset(data_t21, condition!=1))
summary(dist.lm)

# between T-S and M-S
dist.lm = lm(distance ~ condition, data=subset(data_t21, condition!=0))
summary(dist.lm)

